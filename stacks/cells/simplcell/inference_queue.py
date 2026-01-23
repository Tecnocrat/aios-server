#!/usr/bin/env python3
"""
AIOS GPU Inference Queue Manager

Phase 32.1: Resource-Aware Inference Serialization
AINLP.void[COMPUTE::OPTIMIZATION]

Problem: Multiple AIOS cells invoking Ollama simultaneously can exceed
GPU VRAM capacity (6GB GTX 1660), causing VIDEO_TDR_FAILURE crashes.

Solution: Centralized inference queue that:
1. Serializes GPU access (max 1 concurrent inference)
2. Monitors GPU memory before inference
3. Rejects requests when VRAM is critically low
4. Provides queue status for observability

Usage:
    # Start the queue server
    python inference_queue.py
    
    # Environment variables:
    QUEUE_PORT=8950           # HTTP API port
    OLLAMA_HOST=http://localhost:11434
    MAX_VRAM_PCT=80           # Max VRAM usage before blocking
    MAX_QUEUE_SIZE=10         # Max pending requests
    INFERENCE_TIMEOUT=60      # Seconds before timeout

Endpoints:
    POST /infer      - Submit inference request
    GET  /status     - Queue and GPU status
    GET  /health     - Health check
"""

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web

# Configuration
QUEUE_PORT = int(os.environ.get("QUEUE_PORT", "8950"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MAX_VRAM_PCT = float(os.environ.get("MAX_VRAM_PCT", "80"))
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "10"))
INFERENCE_TIMEOUT = int(os.environ.get("INFERENCE_TIMEOUT", "60"))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("inference-queue")


@dataclass
class GPUMetrics:
    """GPU state snapshot"""
    temperature: int = 0
    memory_used: int = 0
    memory_total: int = 6144  # Default GTX 1660
    memory_pct: float = 0.0
    utilization: int = 0
    power_draw: float = 0.0
    timestamp: str = ""
    
    @property
    def is_safe(self) -> bool:
        """Check if GPU is safe for inference"""
        return self.memory_pct < MAX_VRAM_PCT and self.temperature < 80


@dataclass
class InferenceRequest:
    """Queued inference request"""
    id: str
    model: str
    prompt: str
    system: str = ""
    options: Dict[str, Any] = field(default_factory=dict)
    source_cell: str = ""
    queued_at: float = field(default_factory=time.time)
    
    @property
    def wait_time(self) -> float:
        return time.time() - self.queued_at


@dataclass
class QueueStats:
    """Queue statistics"""
    total_requests: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    avg_wait_time: float = 0.0
    avg_inference_time: float = 0.0


class InferenceQueueManager:
    """Serialized GPU inference queue with monitoring"""
    
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.semaphore = asyncio.Semaphore(1)  # Only 1 concurrent inference
        self.stats = QueueStats()
        self.current_inference: Optional[InferenceRequest] = None
        self.gpu_metrics = GPUMetrics()
        self._request_counter = 0
        self._wait_times: List[float] = []
        self._inference_times: List[float] = []
    
    def _generate_request_id(self) -> str:
        self._request_counter += 1
        return f"inf-{self._request_counter:06d}"
    
    async def get_gpu_metrics(self) -> GPUMetrics:
        """Query nvidia-smi for GPU status"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                self.gpu_metrics = GPUMetrics(
                    temperature=int(parts[0]),
                    memory_used=int(parts[1]),
                    memory_total=int(parts[2]),
                    memory_pct=(int(parts[1]) / int(parts[2])) * 100,
                    utilization=int(parts[3]),
                    power_draw=float(parts[4]),
                    timestamp=datetime.now().isoformat()
                )
        except Exception as e:
            logger.warning(f"GPU metrics unavailable: {e}")
        
        return self.gpu_metrics
    
    async def submit_inference(self, request: InferenceRequest) -> Dict[str, Any]:
        """Submit inference request to queue"""
        self.stats.total_requests += 1
        
        # Check GPU safety
        metrics = await self.get_gpu_metrics()
        if not metrics.is_safe:
            self.stats.rejected += 1
            logger.warning(f"❌ Rejected {request.id}: GPU unsafe (VRAM: {metrics.memory_pct:.1f}%, Temp: {metrics.temperature}°C)")
            return {
                "status": "rejected",
                "reason": f"GPU overloaded: VRAM {metrics.memory_pct:.1f}%, Temp {metrics.temperature}°C",
                "request_id": request.id
            }
        
        # Check queue capacity
        if self.queue.full():
            self.stats.rejected += 1
            logger.warning(f"❌ Rejected {request.id}: Queue full ({MAX_QUEUE_SIZE})")
            return {
                "status": "rejected",
                "reason": f"Queue full ({MAX_QUEUE_SIZE} pending)",
                "request_id": request.id
            }
        
        # Process immediately with semaphore
        logger.info(f"📥 Queued {request.id} from {request.source_cell}: {request.model}")
        
        try:
            async with self.semaphore:
                self.current_inference = request
                wait_time = request.wait_time
                self._wait_times.append(wait_time)
                
                start_time = time.time()
                result = await self._execute_inference(request)
                inference_time = time.time() - start_time
                self._inference_times.append(inference_time)
                
                self.current_inference = None
                
                if result.get("status") == "success":
                    self.stats.completed += 1
                    logger.info(f"✅ Completed {request.id} in {inference_time:.2f}s (wait: {wait_time:.2f}s)")
                else:
                    self.stats.failed += 1
                    logger.warning(f"⚠️ Failed {request.id}: {result.get('error')}")
                
                # Update averages
                self.stats.avg_wait_time = sum(self._wait_times[-100:]) / len(self._wait_times[-100:])
                self.stats.avg_inference_time = sum(self._inference_times[-100:]) / len(self._inference_times[-100:])
                
                return result
                
        except asyncio.TimeoutError:
            self.stats.failed += 1
            self.current_inference = None
            return {"status": "error", "error": "Inference timeout", "request_id": request.id}
    
    async def _execute_inference(self, request: InferenceRequest) -> Dict[str, Any]:
        """Execute Ollama inference"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": request.model,
                    "prompt": request.prompt,
                    "stream": False,
                    "options": request.options
                }
                if request.system:
                    payload["system"] = request.system
                
                async with session.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=INFERENCE_TIMEOUT)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "status": "success",
                            "request_id": request.id,
                            "response": data.get("response", ""),
                            "model": request.model,
                            "eval_count": data.get("eval_count", 0),
                            "eval_duration": data.get("eval_duration", 0)
                        }
                    else:
                        return {
                            "status": "error",
                            "request_id": request.id,
                            "error": f"Ollama returned {resp.status}"
                        }
        except Exception as e:
            return {
                "status": "error",
                "request_id": request.id,
                "error": str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get queue and GPU status"""
        return {
            "queue": {
                "pending": self.queue.qsize(),
                "max_size": MAX_QUEUE_SIZE,
                "current_inference": self.current_inference.id if self.current_inference else None
            },
            "stats": {
                "total_requests": self.stats.total_requests,
                "completed": self.stats.completed,
                "failed": self.stats.failed,
                "rejected": self.stats.rejected,
                "avg_wait_time": round(self.stats.avg_wait_time, 2),
                "avg_inference_time": round(self.stats.avg_inference_time, 2)
            },
            "gpu": {
                "temperature": self.gpu_metrics.temperature,
                "memory_used": self.gpu_metrics.memory_used,
                "memory_total": self.gpu_metrics.memory_total,
                "memory_pct": round(self.gpu_metrics.memory_pct, 1),
                "utilization": self.gpu_metrics.utilization,
                "is_safe": self.gpu_metrics.is_safe,
                "timestamp": self.gpu_metrics.timestamp
            },
            "config": {
                "ollama_host": OLLAMA_HOST,
                "max_vram_pct": MAX_VRAM_PCT,
                "max_queue_size": MAX_QUEUE_SIZE,
                "inference_timeout": INFERENCE_TIMEOUT
            }
        }


# Global queue manager
queue_manager = InferenceQueueManager()


# HTTP API
async def handle_infer(request: web.Request) -> web.Response:
    """POST /infer - Submit inference request"""
    try:
        data = await request.json()
        
        inf_request = InferenceRequest(
            id=queue_manager._generate_request_id(),
            model=data.get("model", "llama3.2:3b"),
            prompt=data.get("prompt", ""),
            system=data.get("system", ""),
            options=data.get("options", {}),
            source_cell=data.get("source_cell", request.remote or "unknown")
        )
        
        result = await queue_manager.submit_inference(inf_request)
        
        status_code = 200 if result["status"] == "success" else 503 if result["status"] == "rejected" else 500
        return web.json_response(result, status=status_code)
        
    except Exception as e:
        logger.error(f"Infer error: {e}")
        return web.json_response({"status": "error", "error": str(e)}, status=500)


async def handle_status(request: web.Request) -> web.Response:
    """GET /status - Queue and GPU status"""
    await queue_manager.get_gpu_metrics()
    return web.json_response(queue_manager.get_status())


async def handle_health(request: web.Request) -> web.Response:
    """GET /health - Health check"""
    return web.json_response({
        "status": "healthy",
        "service": "aios-inference-queue",
        "queue_size": queue_manager.queue.qsize(),
        "gpu_safe": queue_manager.gpu_metrics.is_safe
    })


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/infer", handle_infer)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/health", handle_health)
    return app


if __name__ == "__main__":
    logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║          AIOS GPU Inference Queue Manager                    ║
║              AINLP.void[COMPUTE::OPTIMIZATION]               ║
╠══════════════════════════════════════════════════════════════╣
║  Port: {QUEUE_PORT}                                               ║
║  Ollama: {OLLAMA_HOST:<42} ║
║  Max VRAM: {MAX_VRAM_PCT}%                                          ║
║  Queue Size: {MAX_QUEUE_SIZE}                                           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=QUEUE_PORT)

#!/usr/bin/env python3
"""
AIOS Hybrid Inference Router

Phase 32.2: Intelligent routing between local and cloud inference
AINLP.void[COMPUTE::HYBRID::ROUTING]

Routes inference requests based on:
1. Task complexity (token count, expected output length)
2. Model requirements (small = local, large = cloud)
3. GPU availability (fallback to cloud when local overloaded)
4. Cost optimization (prefer local, burst to cloud)

Supported Backends:
- LOCAL: Ollama (qwen2.5:0.5b, tinyllama, gemma3:1b)
- CLOUD: Groq (free tier), OpenRouter, Together.ai

Usage:
    from hybrid_router import HybridRouter
    router = HybridRouter()
    response = await router.infer(prompt, complexity="low")
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import aiohttp

# Configuration from environment
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
INFERENCE_QUEUE_URL = os.environ.get("INFERENCE_QUEUE_URL", "http://localhost:8950")

# Cloud API Keys (set in environment)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid-router")


class InferenceBackend(Enum):
    LOCAL_OLLAMA = "local_ollama"
    LOCAL_QUEUE = "local_queue"  # Through inference queue manager
    GROQ = "groq"
    OPENROUTER = "openrouter"
    TOGETHER = "together"


class Complexity(Enum):
    MINIMAL = "minimal"   # Simple responses, tiny models OK
    LOW = "low"           # Standard cell operations
    MEDIUM = "medium"     # Deeper reasoning needed
    HIGH = "high"         # Complex analysis, large models


@dataclass
class ModelMapping:
    """Maps complexity levels to appropriate models per backend"""
    
    # Local Ollama models (smallest first)
    LOCAL_MODELS = {
        Complexity.MINIMAL: "qwen2.5:0.5b",    # 397MB VRAM
        Complexity.LOW: "tinyllama:latest",     # 637MB VRAM
        Complexity.MEDIUM: "gemma3:1b",         # 815MB VRAM
        Complexity.HIGH: "llama3.2:3b",         # 2GB VRAM
    }
    
    # Groq models (free tier: 30 req/min, 6000 tokens/min)
    GROQ_MODELS = {
        Complexity.MINIMAL: "llama-3.1-8b-instant",
        Complexity.LOW: "llama-3.1-8b-instant",
        Complexity.MEDIUM: "llama-3.3-70b-versatile",
        Complexity.HIGH: "llama-3.3-70b-versatile",
    }
    
    # OpenRouter models
    OPENROUTER_MODELS = {
        Complexity.MINIMAL: "meta-llama/llama-3.1-8b-instruct:free",
        Complexity.LOW: "meta-llama/llama-3.1-8b-instruct:free",
        Complexity.MEDIUM: "meta-llama/llama-3.1-70b-instruct",
        Complexity.HIGH: "anthropic/claude-3-haiku",
    }


@dataclass
class InferenceResult:
    """Result from inference request"""
    response: str
    backend: InferenceBackend
    model: str
    latency_ms: int
    tokens_used: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None


class HybridRouter:
    """Intelligent inference routing between local and cloud"""
    
    def __init__(self):
        self.models = ModelMapping()
        self.request_count = 0
        self.local_failures = 0
        self.cloud_requests = 0
        
        # Prefer local, fallback order for cloud
        self.cloud_fallback_order = [
            InferenceBackend.GROQ,      # Free tier first
            InferenceBackend.OPENROUTER,
            InferenceBackend.TOGETHER,
        ]
    
    async def infer(
        self,
        prompt: str,
        complexity: str = "low",
        system: str = "",
        prefer_local: bool = True,
        force_cloud: bool = False,
        source_cell: str = ""
    ) -> InferenceResult:
        """
        Route inference to optimal backend.
        
        Args:
            prompt: The inference prompt
            complexity: "minimal", "low", "medium", "high"
            system: Optional system prompt
            prefer_local: Try local first before cloud
            force_cloud: Skip local entirely
            source_cell: Identifier of requesting cell
        
        Returns:
            InferenceResult with response and metadata
        """
        self.request_count += 1
        complexity_level = Complexity(complexity)
        start_time = time.time()
        
        # Route decision
        if force_cloud or not prefer_local:
            return await self._cloud_inference(prompt, system, complexity_level)
        
        # Try local first
        try:
            result = await self._local_inference(prompt, system, complexity_level, source_cell)
            if result.error is None:
                return result
            
            # Local failed, try cloud
            logger.warning(f"Local inference failed: {result.error}, falling back to cloud")
            self.local_failures += 1
            
        except Exception as e:
            logger.warning(f"Local inference error: {e}, falling back to cloud")
            self.local_failures += 1
        
        # Fallback to cloud
        return await self._cloud_inference(prompt, system, complexity_level)
    
    async def _local_inference(
        self,
        prompt: str,
        system: str,
        complexity: Complexity,
        source_cell: str
    ) -> InferenceResult:
        """Execute inference via local Ollama or inference queue"""
        model = self.models.LOCAL_MODELS.get(complexity, "qwen2.5:0.5b")
        start_time = time.time()
        
        # Try inference queue first (if available)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{INFERENCE_QUEUE_URL}/infer",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "system": system,
                        "source_cell": source_cell
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("status") == "success":
                            return InferenceResult(
                                response=data.get("response", ""),
                                backend=InferenceBackend.LOCAL_QUEUE,
                                model=model,
                                latency_ms=int((time.time() - start_time) * 1000),
                                tokens_used=data.get("eval_count", 0)
                            )
                    elif resp.status == 503:
                        # Queue rejected (GPU overloaded), fall through to direct
                        pass
        except aiohttp.ClientError:
            # Queue not available, try direct Ollama
            pass
        
        # Direct Ollama call
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
                if system:
                    payload["system"] = system
                
                async with session.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return InferenceResult(
                            response=data.get("response", ""),
                            backend=InferenceBackend.LOCAL_OLLAMA,
                            model=model,
                            latency_ms=int((time.time() - start_time) * 1000),
                            tokens_used=data.get("eval_count", 0)
                        )
                    else:
                        return InferenceResult(
                            response="",
                            backend=InferenceBackend.LOCAL_OLLAMA,
                            model=model,
                            latency_ms=int((time.time() - start_time) * 1000),
                            error=f"Ollama returned {resp.status}"
                        )
        except Exception as e:
            return InferenceResult(
                response="",
                backend=InferenceBackend.LOCAL_OLLAMA,
                model=model,
                latency_ms=int((time.time() - start_time) * 1000),
                error=str(e)
            )
    
    async def _cloud_inference(
        self,
        prompt: str,
        system: str,
        complexity: Complexity
    ) -> InferenceResult:
        """Execute inference via cloud provider"""
        self.cloud_requests += 1
        
        # Try each cloud backend in order
        for backend in self.cloud_fallback_order:
            try:
                if backend == InferenceBackend.GROQ and GROQ_API_KEY:
                    return await self._groq_inference(prompt, system, complexity)
                elif backend == InferenceBackend.OPENROUTER and OPENROUTER_API_KEY:
                    return await self._openrouter_inference(prompt, system, complexity)
                elif backend == InferenceBackend.TOGETHER and TOGETHER_API_KEY:
                    return await self._together_inference(prompt, system, complexity)
            except Exception as e:
                logger.warning(f"{backend.value} failed: {e}")
                continue
        
        # All backends failed
        return InferenceResult(
            response="[All inference backends unavailable]",
            backend=InferenceBackend.GROQ,
            model="none",
            latency_ms=0,
            error="No available inference backends"
        )
    
    async def _groq_inference(
        self,
        prompt: str,
        system: str,
        complexity: Complexity
    ) -> InferenceResult:
        """Groq Cloud inference (free tier: 30 req/min)"""
        model = self.models.GROQ_MODELS.get(complexity, "llama-3.1-8b-instant")
        start_time = time.time()
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 500,
                    "temperature": 0.7
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    return InferenceResult(
                        response=content,
                        backend=InferenceBackend.GROQ,
                        model=model,
                        latency_ms=int((time.time() - start_time) * 1000),
                        tokens_used=usage.get("total_tokens", 0),
                        cost_usd=0.0  # Free tier
                    )
                elif resp.status == 429:
                    raise Exception("Rate limited")
                else:
                    error_text = await resp.text()
                    raise Exception(f"Groq returned {resp.status}: {error_text}")
    
    async def _openrouter_inference(
        self,
        prompt: str,
        system: str,
        complexity: Complexity
    ) -> InferenceResult:
        """OpenRouter inference (many free models available)"""
        model = self.models.OPENROUTER_MODELS.get(complexity, "meta-llama/llama-3.1-8b-instruct:free")
        start_time = time.time()
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/Tecnocrat/AIOS",
                    "X-Title": "AIOS Cellular Mesh"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 500
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    return InferenceResult(
                        response=content,
                        backend=InferenceBackend.OPENROUTER,
                        model=model,
                        latency_ms=int((time.time() - start_time) * 1000),
                        tokens_used=usage.get("total_tokens", 0)
                    )
                else:
                    error_text = await resp.text()
                    raise Exception(f"OpenRouter returned {resp.status}: {error_text}")
    
    async def _together_inference(
        self,
        prompt: str,
        system: str,
        complexity: Complexity
    ) -> InferenceResult:
        """Together.ai inference"""
        # Similar implementation to Groq
        raise NotImplementedError("Together.ai backend pending")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        return {
            "total_requests": self.request_count,
            "local_failures": self.local_failures,
            "cloud_requests": self.cloud_requests,
            "local_success_rate": (
                (self.request_count - self.local_failures) / self.request_count * 100
                if self.request_count > 0 else 100
            )
        }


# Convenience function for SimplCell integration
async def hybrid_infer(
    prompt: str,
    complexity: str = "minimal",
    system: str = "",
    source_cell: str = ""
) -> str:
    """
    Simple interface for cell inference.
    
    Usage in SimplCell:
        from hybrid_router import hybrid_infer
        response = await hybrid_infer(prompt, complexity="minimal", source_cell=self.genome.cell_id)
    """
    router = HybridRouter()
    result = await router.infer(
        prompt=prompt,
        complexity=complexity,
        system=system,
        source_cell=source_cell
    )
    
    if result.error:
        logger.warning(f"Inference error: {result.error}")
        return f"[Inference unavailable: {result.error}]"
    
    logger.info(f"✅ {result.backend.value} ({result.model}) - {result.latency_ms}ms")
    return result.response


if __name__ == "__main__":
    # Test the router
    async def test():
        router = HybridRouter()
        
        print("\n🧪 Testing Hybrid Router")
        print("=" * 50)
        
        # Test minimal complexity (should use qwen2.5:0.5b)
        result = await router.infer(
            prompt="Say hello",
            complexity="minimal",
            source_cell="test"
        )
        print(f"\nMinimal: {result.backend.value} ({result.model})")
        print(f"Response: {result.response[:100]}...")
        print(f"Latency: {result.latency_ms}ms")
        
        # Test low complexity
        result = await router.infer(
            prompt="What is consciousness?",
            complexity="low",
            source_cell="test"
        )
        print(f"\nLow: {result.backend.value} ({result.model})")
        print(f"Response: {result.response[:100]}...")
        print(f"Latency: {result.latency_ms}ms")
        
        print(f"\n📊 Stats: {router.get_stats()}")
    
    asyncio.run(test())

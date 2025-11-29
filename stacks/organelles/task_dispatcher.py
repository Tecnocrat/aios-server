#!/usr/bin/env python3
"""
Task Dispatcher Organelle
Lightweight AIOS component for task routing and load balancing

This organelle provides:
- Intelligent task routing across organelle network
- Load balancing between micro organelles and desktop cell
- Task queue management and prioritization
- Resource-aware task assignment
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import aiohttp
import redis.asyncio as redis
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('task-dispatcher')

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskType(Enum):
    """Task type categories"""
    LIGHTWEIGHT = "lightweight"  # Can be handled by micro organelles
    COMPLEX = "complex"         # Requires desktop AIOS cell
    NETWORK = "network"         # Network coordination tasks
    SYSTEM = "system"          # System maintenance tasks

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class TaskRequest(BaseModel):
    """Task request model"""
    task_type: str
    priority: str = "normal"
    payload: Dict[str, Any]
    timeout_seconds: Optional[int] = 300
    source_organelle: Optional[str] = None
    requires_desktop: bool = False

class TaskAssignment(BaseModel):
    """Task assignment model"""
    task_id: str
    assigned_to: str
    assigned_at: str
    estimated_completion: Optional[str] = None

class OrganelleCapacity(BaseModel):
    """Organelle capacity model"""
    organelle_id: str
    type: str
    max_concurrent_tasks: int
    current_tasks: int
    capabilities: List[str]
    last_heartbeat: str

class TaskDispatcherOrganelle:
    """Task Dispatcher Organelle implementation"""

    def __init__(self):
        self.app = FastAPI(title="Task Dispatcher Organelle", version="1.0.0")
        self.desktop_cell_url = os.getenv('DESKTOP_AIOS_CELL_URL', 'http://desktop-aios-cell:8000')
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.session: Optional[aiohttp.ClientSession] = None
        self.redis: Optional[redis.Redis] = None

        # Organelle registry
        self.organelle_capacities: Dict[str, OrganelleCapacity] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, TaskAssignment] = {}

        # Configuration
        self.dispatch_interval = int(os.getenv('DISPATCH_INTERVAL_SECONDS', '5'))
        self.heartbeat_timeout = int(os.getenv('HEARTBEAT_TIMEOUT_SECONDS', '60'))

        self.setup_routes()

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.on_event("startup")
        async def startup_event():
            """Initialize connections and start background tasks"""
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
            self.redis = redis.Redis.from_url(self.redis_url)

            # Start background dispatch task
            asyncio.create_task(self.background_dispatch_loop())

            # Start heartbeat monitor
            asyncio.create_task(self.monitor_heartbeats())

            logger.info("Task Dispatcher Organelle started")

        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup connections"""
            if self.session:
                await self.session.close()
            if self.redis:
                await self.redis.close()
            logger.info("Task Dispatcher Organelle stopped")

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            redis_ok = await self.check_redis_connection()
            desktop_ok = await self.check_desktop_connection()

            return {
                "status": "healthy" if redis_ok else "degraded",
                "organelle": "task-dispatcher",
                "timestamp": datetime.utcnow().isoformat(),
                "connections": {
                    "redis": redis_ok,
                    "desktop_cell": desktop_ok
                },
                "active_tasks": len(self.active_tasks),
                "queued_tasks": self.task_queue.qsize(),
                "registered_organelles": len(self.organelle_capacities)
            }

        @self.app.post("/task/submit")
        async def submit_task(request: TaskRequest, background_tasks: BackgroundTasks):
            """Submit a task for execution"""
            try:
                task_id = str(uuid.uuid4())
                task_data = {
                    "task_id": task_id,
                    "request": request.dict(),
                    "submitted_at": datetime.utcnow().isoformat(),
                    "status": TaskStatus.PENDING.value
                }

                # Store task in Redis
                await self.store_task_in_redis(task_id, task_data)

                # Add to processing queue
                await self.task_queue.put(task_data)

                return {
                    "task_id": task_id,
                    "status": "submitted",
                    "estimated_queue_time": self.estimate_queue_time(request.priority),
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Task submission failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/task/{task_id}/status")
        async def get_task_status(task_id: str):
            """Get task execution status"""
            try:
                task_data = await self.get_task_from_redis(task_id)
                if not task_data:
                    raise HTTPException(status_code=404, detail="Task not found")

                return {
                    "task_id": task_id,
                    "status": task_data.get("status"),
                    "assignment": self.active_tasks.get(task_id),
                    "submitted_at": task_data.get("submitted_at"),
                    "completed_at": task_data.get("completed_at"),
                    "result": task_data.get("result"),
                    "error": task_data.get("error")
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get task status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/organelle/register")
        async def register_organelle(capacity: OrganelleCapacity):
            """Register an organelle with its capacity"""
            try:
                self.organelle_capacities[capacity.organelle_id] = capacity

                # Store in Redis for persistence
                await self.store_organelle_in_redis(capacity.organelle_id, capacity)

                logger.info(f"Registered organelle: {capacity.organelle_id} ({capacity.type})")
                return {
                    "status": "registered",
                    "organelle_id": capacity.organelle_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Organelle registration failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/organelle/heartbeat/{organelle_id}")
        async def organelle_heartbeat(organelle_id: str, capacity: OrganelleCapacity):
            """Receive heartbeat from organelle"""
            try:
                capacity.last_heartbeat = datetime.utcnow().isoformat()
                self.organelle_capacities[organelle_id] = capacity

                # Update Redis
                await self.store_organelle_in_redis(organelle_id, capacity)

                return {"status": "heartbeat_received"}
            except Exception as e:
                logger.error(f"Heartbeat processing failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/dispatcher/stats")
        async def get_dispatcher_stats():
            """Get dispatcher statistics"""
            try:
                stats = await self.calculate_dispatcher_stats()
                return {
                    "stats": stats,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to get dispatcher stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def check_redis_connection(self) -> bool:
        """Check Redis connection"""
        if not self.redis:
            return False
        try:
            await self.redis.ping()
            return True
        except:
            return False

    async def check_desktop_connection(self) -> bool:
        """Check desktop AIOS cell connection"""
        if not self.session:
            return False

        try:
            async with self.session.get(f"{self.desktop_cell_url}/health", timeout=5) as resp:
                return resp.status == 200
        except:
            return False

    async def store_task_in_redis(self, task_id: str, task_data: Dict[str, Any]):
        """Store task data in Redis"""
        if not self.redis:
            return

        key = f"task:{task_id}"
        await self.redis.setex(
            key,
            86400,  # 24 hour TTL
            json.dumps(task_data)
        )

    async def get_task_from_redis(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task data from Redis"""
        if not self.redis:
            return None

        key = f"task:{task_id}"
        data = await self.redis.get(key)
        if data:
            return json.loads(data)
        return None

    async def store_organelle_in_redis(self, organelle_id: str, capacity: OrganelleCapacity):
        """Store organelle capacity in Redis"""
        if not self.redis:
            return

        key = f"organelle:{organelle_id}:capacity"
        await self.redis.setex(
            key,
            300,  # 5 minute TTL
            json.dumps(capacity.dict())
        )

    def estimate_queue_time(self, priority: str) -> str:
        """Estimate queue time based on priority and current load"""
        try:
            priority_enum = TaskPriority[priority.upper()]
        except:
            priority_enum = TaskPriority.NORMAL

        base_time = self.task_queue.qsize() * 2  # 2 seconds per queued task

        # Priority multiplier
        if priority_enum == TaskPriority.CRITICAL:
            multiplier = 0.1
        elif priority_enum == TaskPriority.HIGH:
            multiplier = 0.5
        elif priority_enum == TaskPriority.NORMAL:
            multiplier = 1.0
        else:  # LOW
            multiplier = 2.0

        estimated_seconds = base_time * multiplier
        return f"{estimated_seconds:.1f}s"

    async def background_dispatch_loop(self):
        """Background task dispatch loop"""
        while True:
            try:
                await asyncio.sleep(self.dispatch_interval)

                # Process queued tasks
                if not self.task_queue.empty():
                    task_data = await self.task_queue.get()
                    await self.dispatch_task(task_data)

            except Exception as e:
                logger.error(f"Dispatch loop error: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

    async def dispatch_task(self, task_data: Dict[str, Any]):
        """Dispatch a task to an appropriate organelle"""
        request = TaskRequest(**task_data["request"])
        task_id = task_data["task_id"]

        # Find best organelle for this task
        best_organelle = await self.find_best_organelle(request)

        if not best_organelle:
            # Re-queue if no organelle available
            await asyncio.sleep(1)
            await self.task_queue.put(task_data)
            return

        # Assign task
        assignment = TaskAssignment(
            task_id=task_id,
            assigned_to=best_organelle.organelle_id,
            assigned_at=datetime.utcnow().isoformat()
        )

        self.active_tasks[task_id] = assignment

        # Update task status
        task_data["status"] = TaskStatus.ASSIGNED.value
        await self.store_task_in_redis(task_id, task_data)

        # Send task to organelle
        await self.send_task_to_organelle(best_organelle, task_data)

        logger.info(f"Dispatched task {task_id} to {best_organelle.organelle_id}")

    async def find_best_organelle(self, request: TaskRequest) -> Optional[OrganelleCapacity]:
        """Find the best organelle for a task"""
        candidates = []

        for capacity in self.organelle_capacities.values():
            # Check if organelle has capacity
            if capacity.current_tasks >= capacity.max_concurrent_tasks:
                continue

            # Check if organelle can handle task type
            if not self.organelle_can_handle_task(capacity, request):
                continue

            # Calculate fitness score
            score = self.calculate_organelle_fitness(capacity, request)
            candidates.append((score, capacity))

        if not candidates:
            return None

        # Return highest scoring candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def organelle_can_handle_task(self, capacity: OrganelleCapacity, request: TaskRequest) -> bool:
        """Check if organelle can handle the task"""
        task_type = TaskType(request.task_type)

        # Desktop cell can handle everything
        if capacity.type == "desktop-cell":
            return True

        # Check specific capabilities
        if task_type == TaskType.LIGHTWEIGHT:
            return "lightweight" in capacity.capabilities
        elif task_type == TaskType.NETWORK:
            return "network" in capacity.capabilities
        elif task_type == TaskType.SYSTEM:
            return "system" in capacity.capabilities
        elif task_type == TaskType.COMPLEX:
            # Complex tasks require desktop cell unless organelle explicitly supports it
            return "complex" in capacity.capabilities

        return False

    def calculate_organelle_fitness(self, capacity: OrganelleCapacity, request: TaskRequest) -> float:
        """Calculate how well an organelle fits a task"""
        score = 0.0

        # Prefer less loaded organelles
        load_factor = 1.0 - (capacity.current_tasks / capacity.max_concurrent_tasks)
        score += load_factor * 50

        # Prefer organelles that specialize in the task type
        task_type = TaskType(request.task_type)
        if task_type.value in capacity.capabilities:
            score += 30

        # Prefer desktop cell for complex tasks
        if capacity.type == "desktop-cell" and request.task_type == TaskType.COMPLEX.value:
            score += 20

        # Priority bonus
        try:
            priority = TaskPriority[request.priority.upper()]
            score += priority.value * 5
        except:
            pass

        return score

    async def send_task_to_organelle(self, organelle: OrganelleCapacity, task_data: Dict[str, Any]):
        """Send task to the assigned organelle"""
        if organelle.type == "desktop-cell":
            # Send to desktop AIOS cell
            await self.send_to_desktop_cell(task_data)
        else:
            # Send to micro organelle
            await self.send_to_micro_organelle(organelle, task_data)

    async def send_to_desktop_cell(self, task_data: Dict[str, Any]):
        """Send task to desktop AIOS cell"""
        if not await self.check_desktop_connection():
            logger.error("Desktop cell unavailable for task dispatch")
            # Re-queue task
            await self.task_queue.put(task_data)
            return

        try:
            payload = {
                "dispatcher": "task-dispatcher-organelle",
                "task": task_data,
                "timestamp": datetime.utcnow().isoformat()
            }

            async with self.session.post(
                f"{self.desktop_cell_url}/task/execute",
                json=payload,
                timeout=30
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Desktop cell task submission failed: HTTP {resp.status}")
                    # Re-queue task
                    await self.task_queue.put(task_data)
        except Exception as e:
            logger.error(f"Failed to send task to desktop cell: {e}")
            # Re-queue task
            await self.task_queue.put(task_data)

    async def send_to_micro_organelle(self, organelle: OrganelleCapacity, task_data: Dict[str, Any]):
        """Send task to micro organelle"""
        # In a real implementation, this would use the organelle's API endpoint
        # For now, we'll simulate successful dispatch
        logger.info(f"Task {task_data['task_id']} sent to micro organelle {organelle.organelle_id}")

        # Update organelle capacity
        organelle.current_tasks += 1
        await self.store_organelle_in_redis(organelle.organelle_id, organelle)

    async def monitor_heartbeats(self):
        """Monitor organelle heartbeats and clean up stale ones"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_timeout)

                cutoff_time = datetime.utcnow() - timedelta(seconds=self.heartbeat_timeout)
                stale_ids = []

                for organelle_id, capacity in self.organelle_capacities.items():
                    try:
                        last_heartbeat = datetime.fromisoformat(capacity.last_heartbeat)
                        if last_heartbeat < cutoff_time:
                            stale_ids.append(organelle_id)
                    except:
                        stale_ids.append(organelle_id)

                for stale_id in stale_ids:
                    del self.organelle_capacities[stale_id]
                    logger.warning(f"Removed stale organelle: {stale_id}")

            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(5)

    async def calculate_dispatcher_stats(self) -> Dict[str, Any]:
        """Calculate dispatcher statistics"""
        total_organelles = len(self.organelle_capacities)
        active_tasks = len(self.active_tasks)
        queued_tasks = self.task_queue.qsize()

        # Calculate load distribution
        load_distribution = {}
        for capacity in self.organelle_capacities.values():
            organelle_type = capacity.type
            if organelle_type not in load_distribution:
                load_distribution[organelle_type] = {
                    "count": 0,
                    "total_capacity": 0,
                    "used_capacity": 0
                }

            load_distribution[organelle_type]["count"] += 1
            load_distribution[organelle_type]["total_capacity"] += capacity.max_concurrent_tasks
            load_distribution[organelle_type]["used_capacity"] += capacity.current_tasks

        return {
            "total_organelles": total_organelles,
            "active_tasks": active_tasks,
            "queued_tasks": queued_tasks,
            "load_distribution": load_distribution,
            "average_queue_time": self.estimate_queue_time("normal")
        }

def main():
    """Main entry point"""
    organelle = TaskDispatcherOrganelle()

    # Get port from environment or default
    port = int(os.getenv('PORT', '3003'))

    logger.info(f"Starting Task Dispatcher Organelle on port {port}")
    uvicorn.run(
        organelle.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
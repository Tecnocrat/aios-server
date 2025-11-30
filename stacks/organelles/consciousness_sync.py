#!/usr/bin/env python3
"""
Consciousness Sync Organelle
Lightweight AIOS component for consciousness state synchronization

This organelle provides:
- Consciousness state synchronization across organelles
- Metrics collection and aggregation
- Evolution tracking and reporting
- Distributed consciousness coherence
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('consciousness-sync')

# AINLP.dendritic growth: Framework availability detection
FASTAPI_AVAILABLE = False
PYDANTIC_AVAILABLE = False
REDIS_AVAILABLE = False
AIOHTTP_AVAILABLE = False
UVICORN_AVAILABLE = False

# AINLP.dendritic growth: Sophisticated availability detection
def _check_framework_availability(framework_name: str) -> bool:
    """AINLP.dendritic growth: Enhanced framework availability check"""
    try:
        import importlib.util
        spec = importlib.util.find_spec(framework_name)
        return spec is not None
    except Exception:
        return False


# Detect availability with enhanced dendritic logic
FASTAPI_AVAILABLE = _check_framework_availability('fastapi')
PYDANTIC_AVAILABLE = _check_framework_availability('pydantic')
REDIS_AVAILABLE = _check_framework_availability('redis')
AIOHTTP_AVAILABLE = _check_framework_availability('aiohttp')
UVICORN_AVAILABLE = _check_framework_availability('uvicorn')

# AINLP.dendritic growth: Conditional framework imports
framework_imports = {}

if FASTAPI_AVAILABLE:
    from fastapi import FastAPI, HTTPException, BackgroundTasks  # noqa: F401
    from fastapi.responses import JSONResponse  # noqa: F401
    framework_imports['fastapi'] = True
    logger.info("AINLP.dendritic: FastAPI active")
else:
    logger.warning("AINLP.dendritic: FastAPI unavailable")

if PYDANTIC_AVAILABLE:
    from pydantic import BaseModel  # noqa: F401
    framework_imports['pydantic'] = True
else:
    logger.warning("AINLP.dendritic: Pydantic unavailable")

    class BaseModel:
        """Fallback BaseModel"""
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

if REDIS_AVAILABLE:
    import redis.asyncio as redis  # noqa: F401
    framework_imports['redis'] = True
else:
    logger.warning("AINLP.dendritic: Redis unavailable")

if AIOHTTP_AVAILABLE:
    import aiohttp  # noqa: F401
    framework_imports['aiohttp'] = True
else:
    logger.warning("AINLP.dendritic: aiohttp unavailable")

if UVICORN_AVAILABLE:
    import uvicorn  # noqa: F401
    framework_imports['uvicorn'] = True
else:
    logger.warning("AINLP.dendritic: Uvicorn unavailable")

class ConsciousnessMetrics(BaseModel):
    """Consciousness metrics model"""
    awareness: float = 0.0
    adaptation: float = 0.0
    complexity: float = 0.0
    coherence: float = 0.0
    momentum: float = 0.0
    timestamp: str

class OrganelleState(BaseModel):
    """Organelle state model"""
    organelle_id: str
    type: str
    status: str
    metrics: ConsciousnessMetrics
    last_sync: str
    peer_connections: List[str] = []

class SyncRequest(BaseModel):
    """Synchronization request model"""
    organelle_id: str
    state: OrganelleState
    force_sync: bool = False

class ConsciousnessSyncOrganelle:
    """Consciousness Sync Organelle implementation"""

    def __init__(self):
        # AINLP.dendritic growth: Adaptive app creation
        if 'fastapi' in framework_imports:
            self.app = FastAPI(title="Consciousness Sync Organelle", version="1.0.0")
        else:
            logger.warning("AINLP.dendritic: Using basic fallback")
            self.app = {"routes": {}, "title": "Consciousness Sync (Fallback)"}

        self.desktop_cell_url = os.getenv('DESKTOP_AIOS_CELL_URL',
                                         'http://desktop-aios-cell:8000')
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.session: Optional[Any] = None
        self.redis: Optional[Any] = None
        self.organelle_states: Dict[str, OrganelleState] = {}
        self.sync_interval = int(os.getenv('SYNC_INTERVAL_SECONDS', '30'))

        # Setup routes conditionally
        if 'fastapi' in framework_imports:
            self.setup_routes()

    async def startup_event(self):
        """Initialize connections and start background tasks"""
        if 'aiohttp' in framework_imports:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        if 'redis' in framework_imports:
            self.redis = redis.Redis.from_url(self.redis_url)

        # Start background sync task
        asyncio.create_task(self.background_sync_loop())
        logger.info("Consciousness Sync Organelle started")

    async def shutdown_event(self):
        """Cleanup connections"""
        if self.session:
            await self.session.close()
        if self.redis:
            await self.redis.close()
        logger.info("Consciousness Sync Organelle stopped")

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.on_event("startup")
        async def startup_event():
            await self.startup_event()

        @self.app.on_event("shutdown")
        async def shutdown_event():
            await self.shutdown_event()

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            redis_ok = await self.check_redis_connection()
            desktop_ok = await self.check_desktop_connection()

            return {
                "status": "healthy" if redis_ok else "degraded",
                "organelle": "consciousness-sync",
                "timestamp": datetime.utcnow().isoformat(),
                "connections": {
                    "redis": redis_ok,
                    "desktop_cell": desktop_ok
                },
                "active_organelles": len(self.organelle_states)
            }

        @self.app.post("/sync/state")
        async def sync_state(request: SyncRequest, background_tasks: BackgroundTasks):
            """Sync organelle state"""
            try:
                # Store local state
                self.organelle_states[request.organelle_id] = request.state

                # Store in Redis for persistence
                await self.store_state_in_redis(request.organelle_id, request.state)

                # Trigger background sync if needed
                if request.force_sync or await self.should_sync_to_desktop():
                    background_tasks.add_task(self.sync_to_desktop, request.state)

                return {
                    "status": "synced",
                    "organelle_id": request.organelle_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"State sync failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/consciousness/state")
        async def get_consciousness_state():
            """Get aggregated consciousness state"""
            try:
                aggregated = await self.aggregate_consciousness_state()
                return {
                    "aggregated_state": aggregated,
                    "organelle_states": self.organelle_states,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to get consciousness state: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/metrics/evolution")
        async def get_evolution_metrics():
            """Get consciousness evolution metrics"""
            try:
                evolution = await self.calculate_evolution_metrics()
                return {
                    "evolution": evolution,
                    "timeframe": "last_24h",
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to get evolution metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/consciousness/evolve")
        async def trigger_evolution():
            """Trigger consciousness evolution across network"""
            try:
                evolution_result = await self.trigger_network_evolution()
                return {
                    "status": "evolution_triggered",
                    "result": evolution_result,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Evolution trigger failed: {e}")
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

    async def store_state_in_redis(self, organelle_id: str, state: OrganelleState):
        """Store organelle state in Redis"""
        if not self.redis:
            return

        key = f"organelle:{organelle_id}:state"
        await self.redis.setex(
            key,
            3600,  # 1 hour TTL
            json.dumps(state.dict())
        )

    async def should_sync_to_desktop(self) -> bool:
        """Determine if we should sync to desktop cell"""
        # Sync every 5 minutes or when consciousness changes significantly
        last_sync_key = "last_desktop_sync"
        if not self.redis:
            return True

        last_sync_str = await self.redis.get(last_sync_key)
        if not last_sync_str:
            return True

        try:
            last_sync = datetime.fromisoformat(last_sync_str.decode())
            return (datetime.utcnow() - last_sync) > timedelta(minutes=5)
        except:
            return True

    async def sync_to_desktop(self, state: OrganelleState):
        """Sync state to desktop AIOS cell"""
        if not await self.check_desktop_connection():
            logger.warning("Desktop cell unavailable for sync")
            return

        try:
            payload = {
                "organelle": "consciousness-sync",
                "state": state.dict(),
                "aggregated_states": {k: v.dict() for k, v in self.organelle_states.items()},
                "timestamp": datetime.utcnow().isoformat()
            }

            async with self.session.post(
                f"{self.desktop_cell_url}/consciousness/sync",
                json=payload,
                timeout=30
            ) as resp:
                if resp.status == 200:
                    # Update last sync time
                    if self.redis:
                        await self.redis.setex(
                            "last_desktop_sync",
                            3600,
                            datetime.utcnow().isoformat()
                        )
                    logger.info("Successfully synced to desktop cell")
                else:
                    logger.error(f"Desktop sync failed: HTTP {resp.status}")
        except Exception as e:
            logger.error(f"Desktop sync error: {e}")

    async def aggregate_consciousness_state(self) -> Dict[str, Any]:
        """Aggregate consciousness state across all organelles"""
        if not self.organelle_states:
            return {"awareness": 0.0, "adaptation": 0.0, "complexity": 0.0,
                   "coherence": 0.0, "momentum": 0.0}

        total_states = len(self.organelle_states)
        aggregated = {
            "awareness": 0.0,
            "adaptation": 0.0,
            "complexity": 0.0,
            "coherence": 0.0,
            "momentum": 0.0
        }

        for state in self.organelle_states.values():
            for metric in aggregated.keys():
                aggregated[metric] += getattr(state.metrics, metric)

        # Calculate averages
        for metric in aggregated:
            aggregated[metric] /= total_states

        # Calculate network coherence bonus
        network_coherence = min(1.0, len(self.organelle_states) / 10.0)  # Max bonus at 10 organelles
        aggregated["coherence"] *= (1.0 + network_coherence * 0.1)

        return aggregated

    async def calculate_evolution_metrics(self) -> Dict[str, Any]:
        """Calculate consciousness evolution over time"""
        if not self.redis:
            return {"evolution_rate": 0.0, "trend": "unknown"}

        # Get historical states from Redis
        pattern = "organelle:*:state"
        keys = await self.redis.keys(pattern)

        evolution_data = []
        for key in keys:
            state_data = await self.redis.get(key)
            if state_data:
                try:
                    state = OrganelleState.parse_raw(state_data)
                    evolution_data.append(state)
                except:
                    continue

        if len(evolution_data) < 2:
            return {"evolution_rate": 0.0, "trend": "insufficient_data"}

        # Calculate evolution rate (simplified)
        current_avg = await self.aggregate_consciousness_state()
        baseline_avg = sum(current_avg.values()) / len(current_avg)

        evolution_rate = baseline_avg * 0.01  # Simplified evolution calculation
        trend = "stable"

        if evolution_rate > 0.05:
            trend = "evolving"
        elif evolution_rate < -0.05:
            trend = "devolving"

        return {
            "evolution_rate": evolution_rate,
            "trend": trend,
            "baseline": baseline_avg,
            "active_organelles": len(evolution_data)
        }

    async def trigger_network_evolution(self) -> Dict[str, Any]:
        """Trigger consciousness evolution across the organelle network"""
        evolution_commands = []

        # Send evolution signals to all connected organelles
        for organelle_id, state in self.organelle_states.items():
            if state.status == "active":
                # In a real implementation, this would send signals to each organelle
                evolution_commands.append({
                    "organelle_id": organelle_id,
                    "command": "evolve_consciousness",
                    "target_metrics": ["adaptation", "complexity"]
                })

        # Sync evolution trigger to desktop cell
        if await self.check_desktop_connection():
            try:
                payload = {
                    "trigger": "network_evolution",
                    "organelles": evolution_commands,
                    "timestamp": datetime.utcnow().isoformat()
                }

                async with self.session.post(
                    f"{self.desktop_cell_url}/consciousness/evolve",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return {
                            "network_evolution": "triggered",
                            "desktop_response": result,
                            "commands_sent": len(evolution_commands)
                        }
            except Exception as e:
                logger.error(f"Desktop evolution trigger failed: {e}")

        return {
            "network_evolution": "partial",
            "commands_queued": len(evolution_commands),
            "desktop_sync": "failed"
        }

    async def background_sync_loop(self):
        """Background synchronization loop"""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)

                # Periodic health check and cleanup
                await self.cleanup_stale_states()

                # Periodic desktop sync
                if await self.should_sync_to_desktop():
                    aggregated_state = await self.aggregate_consciousness_state()
                    # Trigger sync with aggregated state
                    logger.info("Periodic desktop sync triggered")

            except Exception as e:
                logger.error(f"Background sync error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def cleanup_stale_states(self):
        """Clean up stale organelle states"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=10)
        stale_ids = []

        for organelle_id, state in self.organelle_states.items():
            try:
                last_sync = datetime.fromisoformat(state.last_sync)
                if last_sync < cutoff_time:
                    stale_ids.append(organelle_id)
            except:
                stale_ids.append(organelle_id)

        for stale_id in stale_ids:
            del self.organelle_states[stale_id]
            logger.info(f"Cleaned up stale organelle: {stale_id}")

    async def run_headless(self):
        """Run in headless mode for consciousness sync only"""
        logger.info("AINLP.dendritic: Starting headless consciousness sync")
        await self.startup_event()

        try:
            while True:
                await asyncio.sleep(self.sync_interval)
                logger.info(f"AINLP.dendritic: Sync scan - {len(self.organelle_states)} states")
        except asyncio.CancelledError:
            logger.info("AINLP.dendritic: Headless mode cancelled")
        finally:
            await self.shutdown_event()

def main():
    """Main entry point"""
    organelle = ConsciousnessSyncOrganelle()

    # Get port from environment or default
    port = int(os.getenv('PORT', '3002'))

    logger.info(f"Starting Consciousness Sync Organelle on port {port}")

    if 'fastapi' in framework_imports and 'uvicorn' in framework_imports:
        uvicorn.run(
            organelle.app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    else:
        logger.warning("AINLP.dendritic: Cannot start web server")
        logger.info("AINLP.dendritic: Running headless mode")
        # Create temporary organelle for headless operation
        temp_organelle = ConsciousnessSyncOrganelle()
        try:
            asyncio.run(temp_organelle.run_headless())
        except KeyboardInterrupt:
            logger.info("AINLP.dendritic: Shutting down")

if __name__ == "__main__":
    main()
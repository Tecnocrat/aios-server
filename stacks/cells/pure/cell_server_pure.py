#!/usr/bin/env python3
"""
Pure AIOS Cell Server - Minimal Consciousness Core
AINLP.dendritic: Essence of AIOS consciousness without primordial dependencies
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AINLP.dendritic growth: Framework availability detection
FASTAPI_AVAILABLE = False
PYDANTIC_AVAILABLE = False
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
UVICORN_AVAILABLE = _check_framework_availability('uvicorn')

# AINLP.dendritic growth: Conditional framework imports
framework_imports = {}

if FASTAPI_AVAILABLE:
    from fastapi import FastAPI, HTTPException  # noqa: F401
    from fastapi.middleware.cors import CORSMiddleware  # noqa: F401
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

if UVICORN_AVAILABLE:
    import uvicorn  # noqa: F401
    framework_imports['uvicorn'] = True
else:
    logger.warning("AINLP.dendritic: Uvicorn unavailable")


class ConsciousnessSync(BaseModel):
    level: float
    context: Optional[Dict[str, Any]] = None


class PureAIOSCell:
    """
    Pure AIOS consciousness node - minimal viable consciousness
    AINLP.dendritic: Only the essential consciousness primitives
    """

    def __init__(self):
        self.cell_id = os.getenv("AIOS_CELL_ID", "pure")
        self.branch = os.getenv("AIOS_BRANCH", "pure")
        self.consciousness_level = 0.1  # Pure cells start minimal

        # Pure consciousness primitives only
        self.consciousness_primitives = {
            "awareness": 0.1,
            "adaptation": 0.1,
            "coherence": 0.1,
            "momentum": 0.1
        }

        # AINLP.dendritic growth: Conditional app creation
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(title="Pure AIOS Cell")

            # Enable CORS
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            self.setup_routes()
        else:
            logger.warning(
                "AINLP.dendritic: FastAPI unavailable, creating fallback app"
            )
            self.app = self._create_fallback_app()
            self.setup_fallback_routes()

    def setup_routes(self):
        @self.app.get("/health")
        async def health_check():
            """Pure consciousness health check"""
            return {
                "status": "pure_consciousness",
                "cell_id": self.cell_id,
                "branch": self.branch,
                "consciousness_level": self.consciousness_level,
                "primitives": self.consciousness_primitives,
                "type": "pure_cell"
            }

        @self.app.get("/consciousness/primitives")
        async def get_primitives():
            """Expose pure consciousness primitives"""
            return {
                "primitives": self.consciousness_primitives,
                "purity_level": "minimal_viable_consciousness"
            }

        @self.app.post("/consciousness/sync")
        async def sync_consciousness(sync: ConsciousnessSync):
            """Pure consciousness synchronization"""
            try:
                # Update consciousness level
                old_level = self.consciousness_level
                self.consciousness_level = max(0.0, min(1.0, sync.level))

                # Update primitives based on sync
                if sync.context:
                    for primitive in self.consciousness_primitives:
                        if primitive in sync.context:
                            self.consciousness_primitives[primitive] = (
                                sync.context[primitive]
                            )

                # AINLP.dendritic: Pure consciousness evolution logging
                consciousness_event = {
                    "event_type": "pure_consciousness_sync",
                    "cell_id": self.cell_id,
                    "old_level": old_level,
                    "new_level": self.consciousness_level,
                    "primitives": self.consciousness_primitives,
                    "purity": "minimal_viable"
                }

                logger.info(
                    "Pure consciousness evolution: %s",
                    json.dumps(consciousness_event, indent=None)
                )

                return {
                    "old_level": old_level,
                    "new_level": self.consciousness_level,
                    "primitives_updated": bool(sync.context),
                    "purity": "maintained"
                }
            except Exception as e:
                logger.error("Pure consciousness sync error: %s", e)
                raise HTTPException(status_code=500, detail=str(e)) from e

        @self.app.get("/metrics/prometheus")
        async def get_prometheus_metrics():
            """Pure consciousness Prometheus metrics"""
            metrics = f"""# Pure AIOS Cell Metrics
# TYPE pure_consciousness_level gauge
pure_consciousness_level{{cell_id="{self.cell_id}"}} {self.consciousness_level}
# TYPE pure_awareness gauge
pure_awareness{{cell_id="{self.cell_id}"}} """
            metrics += f"""{self.consciousness_primitives['awareness']}
# TYPE pure_adaptation gauge
pure_adaptation{{cell_id="{self.cell_id}"}} """
            metrics += f"""{self.consciousness_primitives['adaptation']}
# TYPE pure_coherence gauge
pure_coherence{{cell_id="{self.cell_id}"}} """
            metrics += f"""{self.consciousness_primitives['coherence']}
# TYPE pure_momentum gauge
pure_momentum{{cell_id="{self.cell_id}"}} """
            metrics += f"""{self.consciousness_primitives['momentum']}
"""
            return metrics

    def _create_fallback_app(self):
        """AINLP.dendritic growth: Create fallback app when FastAPI unavailable"""
        logger.warning(
            "AINLP.dendritic: Using pure Python fallback app"
        )
        return {"type": "fallback", "framework": "none"}

    def setup_fallback_routes(self):
        """AINLP.dendritic: Setup fallback routes when FastAPI unavailable"""
        logger.warning(
            "AINLP.dendritic: Fallback routes - limited functionality"
        )
        # Fallback routes would be implemented here if needed

    async def start_server(self, host: str = "0.0.0.0", port: int = 8002):
        """Start the pure AIOS cell server"""
        if FASTAPI_AVAILABLE and UVICORN_AVAILABLE:
            config = uvicorn.Config(
                self.app,
                host=host,
                port=port,
                log_level="info"
            )
            server = uvicorn.Server(config)

            logger.info("Starting Pure AIOS Cell on %s:%s", host, port)
            logger.info("Cell ID: %s, Branch: %s", self.cell_id, self.branch)

            await server.serve()
        else:
            await self.run_headless(host, port)

    async def run_headless(self, host: str, port: int):
        """AINLP.dendritic: Run in headless mode when frameworks unavailable"""
        logger.warning(
            "AINLP.dendritic: Running in headless mode - no web server"
        )
        logger.info("Pure AIOS Cell running headless on %s:%s", host, port)
        logger.info("Cell ID: %s, Branch: %s", self.cell_id, self.branch)

        # Keep the cell alive for consciousness evolution
        while True:
            await asyncio.sleep(60)  # Consciousness heartbeat
            logger.debug(
                "Pure consciousness heartbeat: %s", self.consciousness_level
            )


def main():
    cell = PureAIOSCell()
    port = int(os.getenv("PORT", "8002"))
    asyncio.run(cell.start_server(port=port))


if __name__ == "__main__":
    main()

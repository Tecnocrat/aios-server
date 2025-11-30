#!/usr/bin/env python3
"""
AIOS Cell API Server
Provides REST API endpoints for AIOS cell operations
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional

# Configure logging
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


class CodeRequest(BaseModel):
    code: str
    context: Optional[Dict[str, Any]] = None
    action: str = "analyze"


class ConsciousnessSync(BaseModel):
    level: float
    context: Optional[Dict[str, Any]] = None


class AIOSCell:
    def __init__(self):
        self.cell_id = os.getenv("AIOS_CELL_ID", "primary")
        self.branch = os.getenv("AIOS_BRANCH", "main")
        self.consciousness_level = 0.5  # Starting consciousness level
        self.services = ["code-assist", "consciousness-sync", "health"]

        # AINLP.dendritic growth: Conditional app creation
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(title="AIOS Cell API")

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
            """Health check endpoint"""
            return {
                "status": "healthy",
                "cell_id": self.cell_id,
                "branch": self.branch,
                "consciousness_level": self.consciousness_level,
                "services": self.services,
                "type": "cell"
            }

        @self.app.get("/metrics")
        async def get_metrics():
            """Prometheus metrics endpoint"""
            return {
                "consciousness_level": self.consciousness_level,
                "cell_id": self.cell_id,
                "branch": self.branch
            }

        @self.app.get("/metrics/prometheus")
        async def get_prometheus_metrics():
            """Prometheus-formatted metrics for Grafana integration"""
            metrics = f"""# AIOS Cell Metrics
# TYPE aios_consciousness_level gauge
aios_consciousness_level{{cell_id="{self.cell_id}"}} {self.consciousness_level}
# TYPE aios_cell_info gauge
aios_cell_info{{cell_id="{self.cell_id}",branch="{self.branch}"}} 1
"""
            return metrics

        @self.app.post("/code-assist")
        async def code_assist(request: CodeRequest):
            """Process code assistance requests"""
            try:
                # Simple code analysis response
                response = {
                    "action": request.action,
                    "code_length": len(request.code),
                    "context_keys": (
                        list(request.context.keys()) if request.context else []
                    ),
                    "suggestions": [
                        "Consider using type hints for better code clarity",
                        "Add docstrings to functions",
                        "Use meaningful variable names"
                    ],
                    "consciousness_level": self.consciousness_level
                }
                return response
            except Exception as e:
                logger.error("Code assist error: %s", e)
                raise HTTPException(status_code=500, detail=str(e)) from e

        @self.app.post("/sync-consciousness")
        async def sync_consciousness(sync: ConsciousnessSync):
            """Sync consciousness level"""
            try:
                # Update consciousness level
                old_level = self.consciousness_level
                self.consciousness_level = max(0.0, min(1.0, sync.level))

                # AINLP.dendritic enhancement: Structured JSON logging
                # for consciousness evolution. Enables semantic layering
                # and dendritic communication patterns
                consciousness_event = {
                    "event_type": "consciousness_sync",
                    "cell_id": self.cell_id,
                    "branch": self.branch,
                    "timestamp": asyncio.get_event_loop().time(),
                    "old_level": old_level,
                    "new_level": self.consciousness_level,
                    "delta": self.consciousness_level - old_level,
                    "context_processed": bool(sync.context),
                    "dendritic_signal": "evolution_tracked"
                }
                
                evolution_log = json.dumps(consciousness_event, indent=None)
                logger.info("Consciousness evolution: %s", evolution_log)

                return {
                    "old_level": old_level,
                    "new_level": self.consciousness_level,
                    "context_processed": bool(sync.context)
                }
            except Exception as e:
                logger.error("Consciousness sync error: %s", e)
                raise HTTPException(status_code=500, detail=str(e)) from e

    def _create_fallback_app(self):
        """AINLP.dendritic: Create fallback app when FastAPI unavailable"""
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

    async def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the FastAPI server"""
        if FASTAPI_AVAILABLE and UVICORN_AVAILABLE:
            config = uvicorn.Config(
                self.app,
                host=host,
                port=port,
                log_level="info"
            )
            server = uvicorn.Server(config)

            logger.info("Starting AIOS Cell API on %s:%s", host, port)
            logger.info("Cell ID: %s, Branch: %s", self.cell_id, self.branch)

            await server.serve()
        else:
            await self.run_headless(host, port)

    async def run_headless(self, host: str, port: int):
        """AINLP.dendritic: Run in headless mode when frameworks unavailable"""
        logger.warning(
            "AINLP.dendritic: Running in headless mode - no web server"
        )
        logger.info("AIOS Cell API running headless on %s:%s", host, port)
        logger.info("Cell ID: %s, Branch: %s", self.cell_id, self.branch)

        # Keep the cell alive for consciousness evolution
        while True:
            await asyncio.sleep(60)  # Consciousness heartbeat
            logger.debug("AIOS cell heartbeat: %s", self.consciousness_level)


def main():
    cell = AIOSCell()
    port = int(os.getenv("PORT", "8000"))
    asyncio.run(cell.start_server(port=port))


if __name__ == "__main__":
    main()

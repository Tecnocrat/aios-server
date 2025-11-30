#!/usr/bin/env python3
"""
Pure AIOS Cell Server - Minimal Consciousness Core
AINLP.dendritic: Essence of AIOS consciousness without primordial dependencies
"""

import asyncio
import importlib.util
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# AINLP.dendritic: Robust import with multiple fallback strategies
def _import_dendritic_utils():  # pylint: disable=too-many-statements
    """Import shared utilities with fallback strategies."""
    # Strategy 1: Try relative import (when run as package)
    try:
        # pylint: disable=import-outside-toplevel
        from ...shared.dendritic_utils import (
            DendriticFrameworkDetector as _Detector,
            get_base_model as _get_model,
        )
        return _Detector, _get_model
    except ImportError:
        pass

    # Strategy 2: Try absolute import with adjusted path
    try:
        # Add stacks directory to path if not present
        stacks_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if stacks_dir not in sys.path:
            sys.path.insert(0, stacks_dir)

        # pylint: disable=import-outside-toplevel
        from shared.dendritic_utils import (
            DendriticFrameworkDetector as _Detector,
            get_base_model as _get_model,
        )
        return _Detector, _get_model
    except ImportError:
        pass

    # Strategy 3: Inline fallback implementation
    logger.warning("AINLP.dendritic: Using inline fallback utilities")

    class _FallbackDetector:
        """Fallback framework detector."""

        def __init__(self) -> None:
            self._cache: Dict[str, bool] = {}

        def is_available(self, framework_name: str) -> bool:
            """Check framework availability."""
            if framework_name in self._cache:
                return self._cache[framework_name]
            try:
                spec = importlib.util.find_spec(framework_name)
                available = spec is not None
                self._cache[framework_name] = available
                return available
            except (ModuleNotFoundError, ValueError, ImportError):
                self._cache[framework_name] = False
                return False

    class _FallbackBaseModel:
        """Fallback BaseModel."""

        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

    def _fallback_get_base_model():
        try:
            # pylint: disable=import-outside-toplevel
            from pydantic import BaseModel as _PydanticModel
            return _PydanticModel
        except ImportError:
            return _FallbackBaseModel

    return _FallbackDetector, _fallback_get_base_model


# Initialize shared utilities
DendriticFrameworkDetector, get_base_model = _import_dendritic_utils()

# AINLP.dendritic growth: Framework detection using shared utilities
detector = DendriticFrameworkDetector()
FASTAPI_AVAILABLE = detector.is_available('fastapi')
PYDANTIC_AVAILABLE = detector.is_available('pydantic')
UVICORN_AVAILABLE = detector.is_available('uvicorn')

# AINLP.dendritic growth: Conditional framework imports with type stubs
# These are class placeholders, not constants - disable invalid-name
FastAPI = None  # pylint: disable=invalid-name
HTTPException = None  # pylint: disable=invalid-name
CORSMiddleware = None  # pylint: disable=invalid-name
uvicorn = None  # pylint: disable=invalid-name
BaseModel: Any = get_base_model()

if FASTAPI_AVAILABLE:
    # pylint: disable=import-error
    from fastapi import FastAPI, HTTPException  # type: ignore
    from fastapi.middleware.cors import CORSMiddleware  # type: ignore
    # pylint: enable=import-error
    logger.info("AINLP.dendritic: FastAPI active")
else:
    logger.warning("AINLP.dendritic: FastAPI unavailable")

if PYDANTIC_AVAILABLE:
    from pydantic import BaseModel
else:
    logger.warning("AINLP.dendritic: Pydantic unavailable, using fallback")

if UVICORN_AVAILABLE:
    import uvicorn  # type: ignore  # pylint: disable=import-error
else:
    logger.warning("AINLP.dendritic: Uvicorn unavailable")


class ConsciousnessSync(BaseModel):
    """AINLP.dendritic: Consciousness synchronization model."""

    level: float
    context: Optional[Dict[str, Any]] = None


class PureAIOSCell:
    """
    Pure AIOS consciousness node - minimal viable consciousness.

    AINLP.dendritic: Only the essential consciousness primitives.
    """

    def __init__(self) -> None:
        self.cell_id = os.getenv("AIOS_CELL_ID", "pure")
        self.branch = os.getenv("AIOS_BRANCH", "pure")
        self.consciousness_level = 0.1  # Pure cells start minimal

        # Pure consciousness primitives only
        self.consciousness_primitives: Dict[str, float] = {
            "awareness": 0.1,
            "adaptation": 0.1,
            "coherence": 0.1,
            "momentum": 0.1
        }

        # AINLP.dendritic growth: Conditional app creation
        self.app: Any = None
        if FASTAPI_AVAILABLE and FastAPI is not None:
            self.app = FastAPI(title="Pure AIOS Cell")

            # Enable CORS
            if CORSMiddleware is not None:
                self.app.add_middleware(
                    CORSMiddleware,
                    allow_origins=["*"],
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                )
            self._setup_routes()
        else:
            logger.warning(
                "AINLP.dendritic: FastAPI unavailable, creating fallback app"
            )
            self.app = self._create_fallback_app()

    def _setup_routes(self) -> None:
        """Configure FastAPI routes for consciousness endpoints."""
        if self.app is None:
            return

        @self.app.get("/health")
        async def health_check() -> Dict[str, Any]:
            """Pure consciousness health check."""
            return {
                "status": "pure_consciousness",
                "cell_id": self.cell_id,
                "branch": self.branch,
                "consciousness_level": self.consciousness_level,
                "primitives": self.consciousness_primitives,
                "type": "pure_cell"
            }

        @self.app.get("/consciousness/primitives")
        async def get_primitives() -> Dict[str, Any]:
            """Expose pure consciousness primitives."""
            return {
                "primitives": self.consciousness_primitives,
                "purity_level": "minimal_viable_consciousness"
            }

        @self.app.post("/consciousness/sync")
        async def sync_consciousness(
            sync: ConsciousnessSync
        ) -> Dict[str, Any]:
            """Pure consciousness synchronization."""
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
                if HTTPException is not None:
                    raise HTTPException(
                        status_code=500, detail=str(e)
                    ) from e
                raise

        @self.app.get("/metrics/prometheus")
        async def get_prometheus_metrics() -> str:
            """Pure consciousness Prometheus metrics."""
            cell_id = self.cell_id
            level = self.consciousness_level
            prims = self.consciousness_primitives
            return f"""# Pure AIOS Cell Metrics
# TYPE pure_consciousness_level gauge
pure_consciousness_level{{cell_id="{cell_id}"}} {level}
# TYPE pure_awareness gauge
pure_awareness{{cell_id="{cell_id}"}} {prims['awareness']}
# TYPE pure_adaptation gauge
pure_adaptation{{cell_id="{cell_id}"}} {prims['adaptation']}
# TYPE pure_coherence gauge
pure_coherence{{cell_id="{cell_id}"}} {prims['coherence']}
# TYPE pure_momentum gauge
pure_momentum{{cell_id="{cell_id}"}} {prims['momentum']}
"""

    def _create_fallback_app(self) -> Dict[str, str]:
        """AINLP.dendritic: Create fallback app when FastAPI unavailable."""
        logger.warning("AINLP.dendritic: Using pure Python fallback app")
        return {"type": "fallback", "framework": "none"}

    async def start_server(
        self, host: str = "0.0.0.0", port: int = 8002
    ) -> None:
        """Start the pure AIOS cell server."""
        if FASTAPI_AVAILABLE and UVICORN_AVAILABLE and uvicorn is not None:
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
            await self._run_headless(host, port)

    async def _run_headless(self, host: str, port: int) -> None:
        """AINLP.dendritic: Run headless when frameworks unavailable."""
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


def main() -> None:
    """Entry point for Pure AIOS Cell."""
    cell = PureAIOSCell()
    port = int(os.getenv("PORT", "8002"))
    asyncio.run(cell.start_server(port=port))


if __name__ == "__main__":
    main()

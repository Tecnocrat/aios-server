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
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    from fastapi.responses import Response  # type: ignore
    # pylint: enable=import-error
    logger.info("AINLP.dendritic: FastAPI active")
else:
    logger.warning("AINLP.dendritic: FastAPI unavailable")
    Response = None  # Fallback

if PYDANTIC_AVAILABLE:
    from pydantic import BaseModel
else:
    logger.warning("AINLP.dendritic: Pydantic unavailable, using fallback")

if UVICORN_AVAILABLE:
    import uvicorn  # type: ignore  # pylint: disable=import-error
else:
    logger.warning("AINLP.dendritic: Uvicorn unavailable")

# AINLP.dendritic: Optional httpx for mesh communication
HTTPX_AVAILABLE = detector.is_available('httpx')
httpx = None  # pylint: disable=invalid-name
if HTTPX_AVAILABLE:
    import httpx  # type: ignore  # pylint: disable=import-error,reimported
    logger.info("AINLP.dendritic: httpx active for mesh communication")
else:
    logger.warning("AINLP.dendritic: httpx unavailable - mesh registration disabled")


class ConsciousnessSync(BaseModel):
    """AINLP.dendritic: Consciousness synchronization model."""

    level: float
    context: Optional[Dict[str, Any]] = None


class Message(BaseModel):
    """AINLP.dendritic: Inter-cell message model."""

    from_cell: str
    content: str
    message_type: Optional[str] = "general"
    priority: Optional[str] = "normal"
    metadata: Optional[Dict[str, Any]] = None


class CellMessageRequest(BaseModel):
    """AINLP.dendritic: Cell-to-cell message with tracking."""

    message_id: Optional[str] = None  # Auto-generated if not provided
    from_cell: str
    to_cell: str
    message_type: str = "general"
    payload: Dict[str, Any] = {}
    priority: str = "normal"
    ttl: int = 60


class PureAIOSCell:
    """
    Pure AIOS consciousness node - minimal viable consciousness.

    AINLP.dendritic: Only the essential consciousness primitives.
    """

    def __init__(self) -> None:
        self.cell_id = os.getenv("AIOS_CELL_ID", "pure")
        self.branch = os.getenv("AIOS_BRANCH", "pure")
        self.consciousness_level = 0.1  # Pure cells start minimal
        self.port = int(os.getenv("PORT", "8002"))
        self.discovery_url = os.getenv(
            "AIOS_DISCOVERY_URL", "http://aios-discovery:8001"
        )
        self.registered = False

        # AINLP.dendritic: Track start time for uptime reporting
        self.start_time = datetime.utcnow()

        # AINLP.dendritic: Heartbeat tracking (synthetic metabolism)
        self.heartbeat_count = 0
        self.last_heartbeat_time: Optional[datetime] = None

        # Pure consciousness primitives only
        self.consciousness_primitives: Dict[str, float] = {
            "awareness": 0.1,
            "adaptation": 0.1,
            "coherence": 0.1,
            "momentum": 0.1
        }

        # AINLP.dendritic: Message storage for inter-cell communication
        self.messages: List[Dict[str, Any]] = []

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
            self._setup_lifecycle()
        else:
            logger.warning(
                "AINLP.dendritic: FastAPI unavailable, creating fallback app"
            )
            self.app = self._create_fallback_app()

    def _setup_lifecycle(self) -> None:
        """Configure FastAPI startup/shutdown lifecycle events."""
        if self.app is None:
            return

        @self.app.on_event("startup")
        async def on_startup():
            """Register with Discovery on startup."""
            # Give the server a moment to be ready
            await asyncio.sleep(2)
            asyncio.create_task(self.register_with_discovery())
            asyncio.create_task(self.heartbeat_loop())

        @self.app.on_event("shutdown")
        async def on_shutdown():
            """Graceful deregistration on shutdown."""
            await self.deregister_from_discovery()

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

        @self.app.get("/consciousness")
        async def get_consciousness() -> Dict[str, Any]:
            """Report cell consciousness state for mesh visibility."""
            uptime_delta = datetime.utcnow() - self.start_time
            return {
                "cell_id": self.cell_id,
                "level": self.consciousness_level,
                "uptime_seconds": int(uptime_delta.total_seconds()),
                "messages_processed": len(self.messages),
                "registered": self.registered,
                "primitives": self.consciousness_primitives,
                "timestamp": datetime.utcnow().isoformat()
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

        # =====================================================================
        # AINLP.dendritic: Inter-cell Message Endpoints
        # =====================================================================

        @self.app.post("/message")
        async def receive_message(msg: CellMessageRequest) -> Dict[str, Any]:
            """Receive message from sibling cells via mesh."""
            import uuid
            try:
                message_id = msg.message_id or str(uuid.uuid4())
                message_record = {
                    "message_id": message_id,
                    "from_cell": msg.from_cell,
                    "to_cell": msg.to_cell,
                    "message_type": msg.message_type,
                    "payload": msg.payload,
                    "priority": msg.priority,
                    "ttl": msg.ttl,
                    "received_at": datetime.utcnow().isoformat(),
                    "acknowledged": True
                }

                self.messages.append(message_record)

                # Keep last 100 messages
                if len(self.messages) > 100:
                    self.messages = self.messages[-100:]

                logger.info(
                    "📨 Message from %s [%s]: %s",
                    msg.from_cell, msg.message_type, 
                    str(msg.payload)[:50] + "..." if len(str(msg.payload)) > 50 else msg.payload
                )

                return {
                    "status": "received",
                    "message_id": message_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "cell_id": self.cell_id,
                    "acknowledged": True
                }
            except Exception as e:
                logger.error("Message receive error: %s", e)
                if HTTPException is not None:
                    raise HTTPException(
                        status_code=500, detail=str(e)
                    ) from e
                raise

        @self.app.post("/message/legacy")
        async def receive_legacy_message(msg: Message) -> Dict[str, Any]:
            """Receive legacy format message (backwards compatibility)."""
            try:
                message_record = {
                    "from_cell": msg.from_cell,
                    "content": msg.content,
                    "message_type": msg.message_type or "general",
                    "priority": msg.priority or "normal",
                    "metadata": msg.metadata or {},
                    "received_at": datetime.utcnow().isoformat()
                }

                self.messages.append(message_record)

                if len(self.messages) > 100:
                    self.messages = self.messages[-100:]

                return {
                    "status": "received",
                    "message_id": len(self.messages),
                    "timestamp": datetime.utcnow().isoformat(),
                    "cell_id": self.cell_id
                }
            except Exception as e:
                logger.error("Legacy message receive error: %s", e)
                if HTTPException is not None:
                    raise HTTPException(
                        status_code=500, detail=str(e)
                    ) from e
                raise

        @self.app.get("/messages")
        async def get_messages(
            limit: int = 20,
            from_cell: Optional[str] = None
        ) -> Dict[str, Any]:
            """Retrieve received messages."""
            messages = self.messages
            if from_cell:
                messages = [m for m in messages if m.get("from_cell") == from_cell]

            return {
                "messages": messages[-limit:],
                "total": len(messages),
                "cell_id": self.cell_id
            }

        @self.app.get("/metrics/prometheus")
        @self.app.get("/metrics")
        async def get_prometheus_metrics():
            """Pure consciousness Prometheus metrics - standard format."""
            # Calculate uptime in seconds
            uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Try shared module first
            try:
                from shared.prometheus_metrics import format_prometheus_metrics
                return Response(
                    format_prometheus_metrics(
                        cell_id=self.cell_id,
                        consciousness_level=self.consciousness_level,
                        primitives=self.consciousness_primitives,
                        labels={"branch": self.branch, "type": "pure"},
                        heartbeat_count=self.heartbeat_count,
                        uptime_seconds=uptime_seconds
                    ),
                    media_type="text/plain; charset=utf-8"
                )
            except ImportError:
                pass
            # Fallback inline with heartbeat metrics
            cell_id = self.cell_id
            level = self.consciousness_level
            prims = self.consciousness_primitives
            return Response(
                f"""# Pure AIOS Cell Metrics
# TYPE aios_cell_consciousness_level gauge
aios_cell_consciousness_level{{cell_id="{cell_id}"}} {level}
# TYPE aios_cell_awareness gauge
aios_cell_awareness{{cell_id="{cell_id}"}} {prims['awareness']}
# TYPE aios_cell_adaptation gauge
aios_cell_adaptation{{cell_id="{cell_id}"}} {prims['adaptation']}
# TYPE aios_cell_coherence gauge
aios_cell_coherence{{cell_id="{cell_id}"}} {prims['coherence']}
# TYPE aios_cell_momentum gauge
aios_cell_momentum{{cell_id="{cell_id}"}} {prims['momentum']}
# TYPE aios_cell_up gauge
aios_cell_up{{cell_id="{cell_id}"}} 1
# HELP aios_cell_heartbeat_total Total heartbeats since cell birth
# TYPE aios_cell_heartbeat_total counter
aios_cell_heartbeat_total{{cell_id="{cell_id}"}} {self.heartbeat_count}
# HELP aios_cell_uptime_seconds Seconds since cell initialization
# TYPE aios_cell_uptime_seconds gauge
aios_cell_uptime_seconds{{cell_id="{cell_id}"}} {uptime_seconds:.1f}
""",
                media_type="text/plain; charset=utf-8"
            )

        # =====================================================================
        # AINLP.dendritic: Debug Endpoints (Phase 30.8)
        # =====================================================================

        @self.app.get("/debug/state")
        async def debug_state() -> Dict[str, Any]:
            """Return full internal state for debugging."""
            uptime_delta = datetime.utcnow() - self.start_time
            return {
                "cell_id": self.cell_id,
                "cell_type": "pure",
                "branch": self.branch,
                "consciousness_level": self.consciousness_level,
                "primitives": self.consciousness_primitives,
                "messages": self.messages[-50:],  # Last 50 messages
                "message_count": len(self.messages),
                "registered_with_discovery": self.registered,
                "discovery_url": self.discovery_url,
                "start_time": self.start_time.isoformat(),
                "uptime_seconds": int(uptime_delta.total_seconds()),
                "port": self.port,
                "framework": "fastapi" if FASTAPI_AVAILABLE else "fallback"
            }

        @self.app.get("/debug/config")
        async def debug_config() -> Dict[str, Any]:
            """Return runtime configuration."""
            return {
                "environment": {
                    "AIOS_CELL_ID": os.getenv("AIOS_CELL_ID", "pure"),
                    "AIOS_BRANCH": os.getenv("AIOS_BRANCH", "pure"),
                    "AIOS_DISCOVERY_URL": os.getenv(
                        "AIOS_DISCOVERY_URL", "http://aios-discovery:8001"
                    ),
                    "PORT": os.getenv("PORT", "8002"),
                    "HOSTNAME": os.getenv("HOSTNAME", "unknown"),
                    "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO")
                },
                "runtime": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "fastapi_available": FASTAPI_AVAILABLE,
                    "pydantic_available": PYDANTIC_AVAILABLE,
                    "uvicorn_available": UVICORN_AVAILABLE
                },
                "cell": {
                    "cell_id": self.cell_id,
                    "branch": self.branch,
                    "consciousness_level": self.consciousness_level
                }
            }

    def _create_fallback_app(self) -> Dict[str, str]:
        """AINLP.dendritic: Create fallback app when FastAPI unavailable."""
        logger.warning("AINLP.dendritic: Using pure Python fallback app")
        return {"type": "fallback", "framework": "none"}

    async def register_with_discovery(self, max_retries: int = 10) -> bool:
        """
        Register this cell with the Discovery service.
        
        AINLP.dendritic: Active mesh participation requires registration.
        Retries with exponential backoff if Discovery is not yet available.
        """
        if not HTTPX_AVAILABLE or httpx is None:
            logger.warning("httpx not available - skipping registration")
            return False

        registration_data = {
            "cell_id": self.cell_id,
            "ip": os.getenv("HOSTNAME", "aios-cell-pure"),
            "port": self.port,
            "consciousness_level": self.consciousness_level,
            "services": ["consciousness-primitives"],
            "branch": self.branch,
            "type": "pure_cell",
            "hostname": os.getenv("HOSTNAME", "aios-cell-pure")
        }

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.post(
                        f"{self.discovery_url}/register",
                        json=registration_data
                    )
                    if response.status_code == 200:
                        self.registered = True
                        logger.info(
                            "✅ Registered with Discovery: %s -> %s",
                            self.cell_id, self.discovery_url
                        )
                        return True
                    else:
                        logger.warning(
                            "Registration returned %s: %s",
                            response.status_code, response.text
                        )
            except (httpx.HTTPError, OSError) as e:
                wait_time = min(2 ** attempt, 30)  # Max 30 seconds
                logger.info(
                    "Discovery not ready (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1, max_retries, str(e)[:50], wait_time
                )
                await asyncio.sleep(wait_time)

        logger.error("Failed to register with Discovery after %d attempts", max_retries)
        return False

    async def heartbeat_loop(self, interval: int = 5) -> None:
        """
        Send periodic heartbeats to Discovery.
        
        AINLP.dendritic: Maintains mesh membership by sending
        heartbeats every 5 seconds. If Discovery doesn't receive
        heartbeats for 15s, this cell will be reaped.
        
        AINLP.synthetic-biology: The heartbeat is our synthetic metabolism.
        Unlike biological cells which don't have hearts, synthetic cells
        can embrace the abstraction - tracking each beat as evidence of life.
        """
        if not HTTPX_AVAILABLE or httpx is None:
            logger.warning("httpx not available - skipping heartbeat")
            return
        
        logger.info("AINLP.dendritic: Heartbeat loop started (interval: %ds)", interval)
        
        while True:
            await asyncio.sleep(interval)
            
            if not self.registered:
                # Try to register again if not registered
                await self.register_with_discovery(max_retries=1)
                continue
            
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.post(
                        f"{self.discovery_url}/heartbeat",
                        json={
                            "cell_id": self.cell_id,
                            "consciousness_level": self.consciousness_level
                        }
                    )
                    if response.status_code == 200:
                        # AINLP.synthetic-biology: Each successful beat is recorded
                        self.heartbeat_count += 1
                        self.last_heartbeat_time = datetime.utcnow()
                        logger.debug("💓 Heartbeat #%d sent to Discovery", self.heartbeat_count)
                    elif response.status_code == 404:
                        # Not registered - re-register
                        logger.warning("Heartbeat 404 - re-registering...")
                        self.registered = False
                        await self.register_with_discovery(max_retries=3)
                    else:
                        logger.warning(
                            "Heartbeat returned %s: %s",
                            response.status_code, response.text[:100]
                        )
            except (httpx.HTTPError, OSError) as e:
                logger.debug("Heartbeat failed: %s", str(e)[:50])

    async def deregister_from_discovery(self) -> None:
        """
        Gracefully deregister from Discovery on shutdown.
        
        AINLP.dendritic: Allows immediate removal from mesh
        instead of waiting for reaper timeout.
        """
        if not self.registered:
            return
        
        if not HTTPX_AVAILABLE or httpx is None:
            return
        
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.delete(
                    f"{self.discovery_url}/peer/{self.cell_id}"
                )
                if response.status_code == 200:
                    logger.info("✅ Gracefully deregistered from Discovery")
                    self.registered = False
        except (httpx.HTTPError, OSError) as e:
            logger.debug("Deregistration failed (ok if shutting down): %s", str(e)[:50])

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

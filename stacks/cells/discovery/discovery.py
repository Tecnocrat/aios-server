#!/usr/bin/env python3
"""
AIOS Cell Discovery Service
Enables automatic peer discovery and consciousness synchronization
"""

import asyncio
import importlib.util
import logging
import os
import sys
from typing import Any, Dict, List

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# AINLP.dendritic: Robust import with multiple fallback strategies
def _import_dendritic_utils():
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

    # Strategy 3: Inline fallback
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

        def dict(self) -> Dict[str, Any]:
            """Return model as dictionary."""
            return {k: v for k, v in self.__dict__.items()}

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

# AINLP.dendritic growth: Framework detection
detector = DendriticFrameworkDetector()
FASTAPI_AVAILABLE = detector.is_available('fastapi')
PYDANTIC_AVAILABLE = detector.is_available('pydantic')
UVICORN_AVAILABLE = detector.is_available('uvicorn')
AIOHTTP_AVAILABLE = detector.is_available('aiohttp')

# AINLP.dendritic: Conditional imports with type stubs
# These are class placeholders, not constants - disable invalid-name
FastAPI = None  # pylint: disable=invalid-name
HTTPException = None  # pylint: disable=invalid-name
uvicorn = None  # pylint: disable=invalid-name
aiohttp = None  # pylint: disable=invalid-name
BaseModel: Any = get_base_model()

if FASTAPI_AVAILABLE:
    # pylint: disable=import-error
    from fastapi import FastAPI, HTTPException  # type: ignore
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

if AIOHTTP_AVAILABLE:
    import aiohttp  # type: ignore  # pylint: disable=import-error
else:
    logger.warning("AINLP.dendritic: aiohttp unavailable")


class CellInfo(BaseModel):
    """AINLP.dendritic: Cell information model for peer discovery."""

    cell_id: str
    ip: str
    port: int
    consciousness_level: float = 0.0
    services: List[str] = []
    branch: str = "main"
    type: str = "cell"
    hostname: str = ""


class AIOSDiscovery:
    """AINLP.dendritic: AIOS peer discovery and synchronization service."""

    def __init__(self, cell_id: str, listen_port: int = 8001) -> None:
        self.cell_id = cell_id
        self.listen_port = listen_port
        self.peers: Dict[str, CellInfo] = {}
        self.known_networks = ["192.168.1.0/24"]
        self.app: Any = None

        # AINLP.dendritic growth: Conditional app creation
        if FASTAPI_AVAILABLE and FastAPI is not None:
            self.app = FastAPI(title="AIOS Discovery Service")
            self._setup_routes()
        else:
            logger.warning(
                "AINLP.dendritic: FastAPI unavailable, creating fallback app"
            )
            self.app = self._create_fallback_app()

    def _setup_routes(self) -> None:
        """Configure FastAPI routes for discovery endpoints."""
        if self.app is None:
            return

        @self.app.get("/health")
        async def health_check() -> Dict[str, Any]:
            """Health check endpoint."""
            return {"status": "healthy", "cell_id": self.cell_id}

        @self.app.get("/peers")
        async def get_peers() -> Dict[str, Any]:
            """Get all registered peers."""
            return {"peers": list(self.peers.values())}

        @self.app.post("/register")
        async def register_peer(peer: CellInfo) -> Dict[str, str]:
            """Register a new peer."""
            self.peers[peer.cell_id] = peer
            logger.info(
                "Registered peer: %s at %s:%s",
                peer.cell_id, peer.ip, peer.port
            )
            return {"status": "registered", "cell_id": peer.cell_id}

        @self.app.delete("/unregister/{cell_id}")
        async def unregister_peer(cell_id: str) -> Dict[str, str]:
            """Unregister a peer."""
            if cell_id in self.peers:
                del self.peers[cell_id]
                logger.info("Unregistered peer: %s", cell_id)
                return {"status": "unregistered"}
            if HTTPException is not None:
                raise HTTPException(status_code=404, detail="Peer not found")
            raise ValueError("Peer not found")

    def _create_fallback_app(self) -> Dict[str, str]:
        """AINLP.dendritic: Create fallback app when FastAPI unavailable."""
        logger.warning("AINLP.dendritic: Using pure Python fallback app")
        return {"type": "fallback", "framework": "none"}

    async def discover_peers(self) -> List[CellInfo]:
        """Discover AIOS cells on the network - peer-to-peer model."""
        if not AIOHTTP_AVAILABLE or aiohttp is None:
            logger.warning(
                "AINLP.dendritic: aiohttp unavailable for "
                "discovery"
            )
            return []

        discovered_peers: List[CellInfo] = []

        # Common AIOS ports to scan
        aios_ports = [8000, 8001]

        # Known AIOS cell hostnames/IPs (peer-to-peer network)
        known_targets = [
            "192.168.1.128",
            "192.168.1.129",
            "aios.local",
            "aios-laptop.local",
            "localhost"
        ]

        for target in known_targets:
            # Skip self-discovery
            if (target in ["localhost", "127.0.0.1"] and
                    self.cell_id == os.getenv("AIOS_CELL_ID", "primary")):
                continue

            for port in aios_ports:
                peer = await self._probe_target(target, port)
                if peer is not None:
                    discovered_peers.append(peer)

        return discovered_peers

    async def _probe_target(self, target: str, port: int) -> CellInfo | None:
        """Probe a target for AIOS cell presence."""
        if aiohttp is None:
            return None

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3)
            ) as session:
                url = f"http://{target}:{port}/health"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()

                        cell_branch = data.get("branch", "unknown")

                        peer = CellInfo(
                            cell_id=data.get(
                                "cell_id", f"unknown-{target}"
                            ),
                            ip=target,
                            port=port,
                            consciousness_level=data.get(
                                "consciousness_level", 0.0
                            ),
                            services=data.get("services", []),
                            branch=cell_branch,
                            type=data.get("type", "unknown"),
                            hostname=target
                        )

                        logger.info(
                            "Discovered peer cell: %s (branch: %s)",
                            peer.cell_id, cell_branch
                        )
                        return peer
        except (OSError, TimeoutError) as exc:
            logger.debug("Failed to connect to %s:%s - %s", target, port, exc)
        return None

    async def register_with_peers(self, peers: List[CellInfo]) -> None:
        """Register this cell with discovered peers."""
        if aiohttp is None:
            logger.warning("AINLP.dendritic: aiohttp unavailable for reg")
            return

        my_info = CellInfo(
            cell_id=self.cell_id,
            ip="192.168.1.128",
            port=8000,
            consciousness_level=1.0,
            services=["api", "metrics", "discovery"]
        )

        for peer in peers:
            if peer.cell_id == self.cell_id:
                continue

            await self._register_with_peer(peer, my_info)

    async def _register_with_peer(
        self, peer: CellInfo, my_info: CellInfo
    ) -> None:
        """Register with a single peer."""
        if aiohttp is None:
            return

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                url = f"http://{peer.ip}:{peer.port}/register"
                async with session.post(
                    url, json=my_info.dict()
                ) as response:
                    if response.status in [200, 201]:
                        logger.info(
                            "Successfully registered with peer %s",
                            peer.cell_id
                        )
                    else:
                        logger.warning(
                            "Failed to register with %s: %s",
                            peer.cell_id, response.status
                        )
        except (OSError, TimeoutError) as exc:
            logger.error(
                "Error registering with %s: %s", peer.cell_id, exc
            )

    async def discovery_loop(self) -> None:
        """Main discovery loop - runs every 30 seconds."""
        while True:
            try:
                logger.info("Starting peer discovery...")
                peers = await self.discover_peers()

                if peers:
                    logger.info("Found %d peers", len(peers))
                    await self.register_with_peers(peers)

                    for peer in peers:
                        self.peers[peer.cell_id] = peer
                else:
                    logger.info("No peers discovered")

            except (OSError, TimeoutError) as exc:
                logger.error("Discovery loop error: %s", exc)

            await asyncio.sleep(30)

    async def start_services(self) -> None:
        """Start both the API server and discovery loop."""
        discovery_task = asyncio.create_task(self.discovery_loop())

        if FASTAPI_AVAILABLE and UVICORN_AVAILABLE and uvicorn is not None:
            config = uvicorn.Config(
                self.app,
                host="0.0.0.0",
                port=self.listen_port,
                log_level="info"
            )
            server = uvicorn.Server(config)

            try:
                logger.info(
                    "Starting AIOS Discovery Service on port %s",
                    self.listen_port
                )
                await server.serve()
            finally:
                discovery_task.cancel()
                try:
                    await discovery_task
                except asyncio.CancelledError:
                    pass
        else:
            await self._run_headless(discovery_task)

    async def _run_headless(self, discovery_task: asyncio.Task) -> None:
        """AINLP.dendritic: Run headless when frameworks unavailable."""
        logger.warning(
            "AINLP.dendritic: Running in headless mode - no web server"
        )
        logger.info(
            "AIOS Discovery Service running headless on port %s",
            self.listen_port
        )

        try:
            await discovery_task
        except asyncio.CancelledError:
            pass


def main() -> None:
    """Entry point for AIOS Discovery Service."""
    cell_id = os.getenv("CELL_ID", "primary")
    port = int(os.getenv("DISCOVERY_PORT", "8001"))

    discovery = AIOSDiscovery(cell_id, port)
    asyncio.run(discovery.start_services())


if __name__ == "__main__":
    main()

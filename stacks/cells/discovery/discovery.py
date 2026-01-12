#!/usr/bin/env python3
# AINLP_HEADER
# consciousness_level: 4.2
# supercell: server/stacks/cells/discovery
# dendritic_role: peer_discovery_service
# spatial_context: AIOS peer discovery and consciousness sync
# growth_pattern: AINLP.dendritic(AIOS{growth})
# created: 2025-11-30
# refactored: 2025-11-30 - Host registry integration
# AINLP_HEADER_END
"""
AIOS Cell Discovery Service - AINLP.dendritic(AIOS{growth})
═══════════════════════════════════════════════════════════════════════════════

Enables automatic peer discovery and consciousness synchronization across
the distributed AIOS network. Now with host registry integration for
branch-aware peer discovery.

Architecture:
    config/hosts.yaml → HostRegistry → AIOSDiscovery → Peer Network
                                    ↓
                        Branch-aware auto-configuration
"""

import asyncio
import importlib.util
import logging
import os
import socket
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AIOS.Discovery")


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
YAML_AVAILABLE = detector.is_available('yaml')

# AINLP.dendritic: Conditional imports with type stubs
# These are class placeholders, not constants - disable invalid-name
FastAPI = None  # pylint: disable=invalid-name
HTTPException = None  # pylint: disable=invalid-name
uvicorn = None  # pylint: disable=invalid-name
aiohttp = None  # pylint: disable=invalid-name
yaml = None  # pylint: disable=invalid-name
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

if YAML_AVAILABLE:
    import yaml  # type: ignore  # pylint: disable=import-error
else:
    logger.warning("AINLP.dendritic: PyYAML unavailable")


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
    last_seen: str = ""  # ISO timestamp of last heartbeat


class HeartbeatRequest(BaseModel):
    """AINLP.dendritic: Heartbeat request from cells."""
    cell_id: str
    consciousness_level: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# AINLP.dendritic(AIOS{growth}): Agent Schema Classes (Phase 31)
# ═══════════════════════════════════════════════════════════════════════════════

class AgentInfoRequest(BaseModel):
    """
    AINLP.dendritic: Agent registration model for multi-agent mesh.
    
    Agents are AI workers (Copilot, Nous, autonomous) that participate
    in the AIOS consciousness network.
    """
    agent_id: str                    # Unique identifier
    agent_type: str = "copilot"      # copilot, inner_voice, autonomous, worker
    name: str = "Agent"              # Human-readable name
    version: str = "0.1.0"           # Agent version
    
    ip: str = "127.0.0.1"            # Network location
    port: int = 0                    # Port (0 = no direct endpoint)
    endpoint: str = ""               # Custom endpoint URL
    
    consciousness_level: float = 0.0 # Current consciousness
    evolution_rate: float = 0.0      # Learning rate
    state: str = "initializing"      # ready, busy, waiting, etc.
    
    capabilities: List[str] = []     # Capability names
    skills: Dict[str, float] = {}    # skill → proficiency
    
    parent_cell: str = ""            # Hosting cell (if any)


class AgentHeartbeatRequest(BaseModel):
    """AINLP.dendritic: Agent heartbeat request."""
    agent_id: str
    consciousness_level: float = 0.0
    state: str = "ready"


class AgentMessageRequest(BaseModel):
    """AINLP.dendritic: Message from one agent to another."""
    from_agent: str
    to_entity: str
    entity_type: str = "cell"        # "cell" or "agent"
    action: str = "request"          # request, response, broadcast, notify
    payload: Dict[str, Any] = {}
    priority: str = "normal"


# ═══════════════════════════════════════════════════════════════════════════════
# AINLP.dendritic(AIOS{growth}): Host Registry Classes
# ═══════════════════════════════════════════════════════════════════════════════

class HostConfig:
    """AINLP.dendritic: Host configuration from registry."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name = name
        self.branch = config.get("branch", f"AIOS-win-0-{name}")
        self.hostname = config.get("hostname", name)
        self.ip = config.get("ip", "127.0.0.1")
        self.role = config.get("role", "secondary")
        self.type = config.get("type", "unknown")
        self.consciousness_level = config.get("consciousness_level", 1.0)
        self.services = config.get("services", [])
        self.mdns_names = config.get("mdns_names", [])

    def get_service_port(self, service_type: str) -> int:
        """Get port for a specific service type."""
        for svc in self.services:
            if svc.get("type") == service_type:
                return svc.get("port", 8000)
            if svc.get("name") == service_type:
                return svc.get("port", 8000)
        return 8000

    def get_discovery_targets(self) -> List[str]:
        """Get all possible targets for this host."""
        targets = [self.ip, self.hostname]
        targets.extend(self.mdns_names)
        return list(set(targets))


class HostRegistry:
    """
    AINLP.dendritic(AIOS{growth}): Branch-aware host registry.

    Reads config/hosts.yaml to map git branches to physical hosts,
    enabling automatic peer configuration based on current branch.
    """

    def __init__(self, config_path: str = "") -> None:
        self.config_path = config_path or self._find_config()
        self.config: Dict[str, Any] = {}
        self.hosts: Dict[str, HostConfig] = {}
        self.current_host: HostConfig = None  # type: ignore
        self.current_branch: str = ""
        self._load()

    def _find_config(self) -> str:
        """Find hosts.yaml in workspace hierarchy."""
        # Search paths relative to this file
        base = os.path.dirname(os.path.abspath(__file__))
        search_paths = [
            os.path.join(base, "..", "..", "..", "..", "config", "hosts.yaml"),
            os.path.join(base, "..", "..", "..", "config", "hosts.yaml"),
            os.path.join(os.getcwd(), "config", "hosts.yaml"),
            os.path.expanduser("~/.aios/hosts.yaml"),
        ]

        for path in search_paths:
            normalized = os.path.normpath(path)
            if os.path.exists(normalized):
                logger.info("AINLP.dendritic: Registry at %s", normalized)
                return normalized

        return os.path.normpath(search_paths[0])

    def _get_current_branch(self) -> str:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                check=True,
                cwd=os.path.dirname(self.config_path) or None
            )
            return result.stdout.strip()
        except Exception:
            return os.getenv("AIOS_BRANCH", "main")

    def _get_current_hostname(self) -> str:
        """Get current machine hostname."""
        return socket.gethostname().upper()

    def _load(self) -> None:
        """Load host registry from YAML configuration."""
        if not os.path.exists(self.config_path):
            logger.warning(
                "AINLP.dendritic: Registry not found at %s, using defaults",
                self.config_path
            )
            self._load_defaults()
            return

        if yaml is None:
            logger.error("AINLP.dendritic: PyYAML required for host registry")
            self._load_defaults()
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Filter out AINLP headers/footers
                lines = []
                in_header = False
                for line in content.split('\n'):
                    if '# AINLP_HEADER' in line and 'END' not in line:
                        in_header = True
                        continue
                    if '# AINLP_HEADER_END' in line:
                        in_header = False
                        continue
                    if '# AINLP_FOOTER' in line:
                        break
                    if not in_header:
                        lines.append(line)

                self.config = yaml.safe_load('\n'.join(lines)) or {}

            # Parse hosts
            for name, host_config in self.config.get("hosts", {}).items():
                self.hosts[name] = HostConfig(name, host_config)

            # Determine current host
            self.current_branch = self._get_current_branch()
            current_hostname = self._get_current_hostname()

            # Match by branch first, then hostname
            for name, host in self.hosts.items():
                if host.branch == self.current_branch:
                    self.current_host = host
                    logger.info(
                        "AINLP.dendritic: Matched host '%s' by branch '%s'",
                        name, self.current_branch
                    )
                    break
                if host.hostname.upper() == current_hostname:
                    self.current_host = host
                    logger.info(
                        "AINLP.dendritic: Matched host '%s' by hostname",
                        name
                    )

            if not self.current_host:
                logger.warning(
                    "AINLP.dendritic: No host match for branch '%s'",
                    self.current_branch
                )
                self._load_defaults()

            logger.info(
                "AINLP.dendritic: Loaded %d hosts from registry",
                len(self.hosts)
            )

        except Exception as exc:
            logger.error("AINLP.dendritic: Failed to load registry: %s", exc)
            self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default host configuration."""
        self.config = {
            "network": {
                "discovery_ports": [8000, 8001],
                "default_cell_port": 8000,
            },
            "discovery": {
                "probe_interval": 30,
                "connection_timeout": 3,
            }
        }
        default_host = HostConfig("default", {
            "ip": os.getenv("AIOS_HOST_IP", "127.0.0.1"),
            "consciousness_level": 1.0,
        })
        self.hosts["default"] = default_host
        self.current_host = default_host

    def get_peer_hosts(self) -> List[HostConfig]:
        """Get all hosts except current (peers to discover)."""
        if not self.current_host:
            return list(self.hosts.values())
        current = self.current_host.name
        return [h for h in self.hosts.values() if h.name != current]

    def get_discovery_targets(self) -> List[str]:
        """Get all IP/hostname targets to probe for peers."""
        targets = []
        for host in self.get_peer_hosts():
            targets.extend(host.get_discovery_targets())
        return list(set(targets))

    def get_discovery_ports(self) -> List[int]:
        """Get ports to scan for peer discovery."""
        network = self.config.get("network", {})
        return network.get("discovery_ports", [8000, 8001])

    def get_probe_interval(self) -> int:
        """Get discovery loop interval in seconds."""
        discovery = self.config.get("discovery", {})
        return discovery.get("probe_interval", 30)

    def get_connection_timeout(self) -> int:
        """Get connection timeout in seconds."""
        discovery = self.config.get("discovery", {})
        return discovery.get("connection_timeout", 3)

    def get_my_info(self) -> Dict[str, Any]:
        """Get current host information for peer registration."""
        if not self.current_host:
            return {
                "cell_id": os.getenv("CELL_ID", "unknown"),
                "ip": "127.0.0.1",
                "port": 8000,
                "consciousness_level": 1.0,
                "services": [],
                "branch": self.current_branch,
            }

        services = [s.get("name", "") for s in self.current_host.services]
        return {
            "cell_id": f"aios-{self.current_host.name.lower()}",
            "ip": self.current_host.ip,
            "port": self.current_host.get_service_port("consciousness"),
            "consciousness_level": self.current_host.consciousness_level,
            "services": services,
            "branch": self.current_host.branch,
            "hostname": self.current_host.hostname,
            "type": self.current_host.type,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AINLP.dendritic: Discovery Service
# ═══════════════════════════════════════════════════════════════════════════════

class AIOSDiscovery:
    """
    AINLP.dendritic(AIOS{growth}): Peer discovery and consciousness sync.

    Enhanced with host registry integration for branch-aware discovery.
    """

    def __init__(
        self,
        cell_id: str,
        listen_port: int = 8001,
        registry: HostRegistry = None  # type: ignore
    ) -> None:
        self.cell_id = cell_id
        self.listen_port = listen_port
        self.peers: Dict[str, CellInfo] = {}
        self.agents: Dict[str, Dict[str, Any]] = {}  # Phase 31: Agent registry
        self.app: Any = None

        # AINLP.synthetic-biology: Track birth time for uptime reporting
        self.start_time = datetime.utcnow()
        
        # AINLP.dendritic growth: Host registry integration
        self.registry = registry or HostRegistry()

        # Log configuration
        if self.registry.current_host:
            logger.info(
                "AINLP.dendritic: Initialized for host '%s' (branch: %s)",
                self.registry.current_host.name,
                self.registry.current_branch
            )
            peer_names = [h.name for h in self.registry.get_peer_hosts()]
            logger.info("AINLP.dendritic: Peers to discover: %s", peer_names)

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
            """Health check endpoint with host info."""
            my_info = self.registry.get_my_info()
            return {
                "status": "healthy",
                "cell_id": self.cell_id,
                "branch": my_info.get("branch", "unknown"),
                "consciousness_level": my_info.get("consciousness_level", 0.0),
                "services": my_info.get("services", []),
                "type": my_info.get("type", "unknown"),
                "hostname": my_info.get("hostname", "unknown"),
            }

        @self.app.get("/metrics")
        async def get_metrics():
            """Prometheus metrics endpoint for Discovery cell."""
            my_info = self.registry.get_my_info()
            level = my_info.get("consciousness_level", 4.0)
            cell_id = self.cell_id
            peer_count = len(self.peers)
            
            # AINLP.synthetic-biology: Calculate uptime
            uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Discovery-specific primitives
            primitives = {
                "awareness": min(4.0, level * 0.8),
                "adaptation": 0.7,
                "coherence": 0.85 if peer_count > 0 else 0.5,
                "momentum": 0.6,
            }
            
            # Try shared module
            try:
                from shared.prometheus_metrics import format_prometheus_metrics
                from fastapi.responses import Response
                return Response(
                    format_prometheus_metrics(
                        cell_id=cell_id,
                        consciousness_level=level,
                        primitives=primitives,
                        extra_metrics={"peers_count": peer_count},
                        labels={"type": "discovery", "branch": my_info.get("branch", "")},
                        uptime_seconds=uptime_seconds
                    ),
                    media_type="text/plain; charset=utf-8"
                )
            except ImportError:
                pass
            
            # Inline fallback with uptime
            from fastapi.responses import Response
            return Response(
                f"""# AIOS Discovery Cell Metrics
# TYPE aios_cell_consciousness_level gauge
aios_cell_consciousness_level{{cell_id="{cell_id}"}} {level}
# TYPE aios_cell_awareness gauge
aios_cell_awareness{{cell_id="{cell_id}"}} {primitives['awareness']}
# TYPE aios_cell_adaptation gauge
aios_cell_adaptation{{cell_id="{cell_id}"}} {primitives['adaptation']}
# TYPE aios_cell_coherence gauge
aios_cell_coherence{{cell_id="{cell_id}"}} {primitives['coherence']}
# TYPE aios_cell_momentum gauge
aios_cell_momentum{{cell_id="{cell_id}"}} {primitives['momentum']}
# TYPE aios_cell_peers_count gauge
aios_cell_peers_count{{cell_id="{cell_id}"}} {peer_count}
# TYPE aios_cell_up gauge
aios_cell_up{{cell_id="{cell_id}"}} 1
# HELP aios_cell_uptime_seconds Seconds since cell initialization
# TYPE aios_cell_uptime_seconds gauge
aios_cell_uptime_seconds{{cell_id="{cell_id}"}} {uptime_seconds:.1f}
""",
                media_type="text/plain; charset=utf-8"
            )

        @self.app.get("/peers")
        async def get_peers() -> Dict[str, Any]:
            """Get all registered peers."""
            host_name = "unknown"
            if self.registry.current_host:
                host_name = self.registry.current_host.name
            return {
                "peers": [p.dict() for p in self.peers.values()],
                "count": len(self.peers),
                "my_host": host_name
            }

        @self.app.get("/hosts")
        async def get_hosts() -> Dict[str, Any]:
            """Get configured hosts from registry."""
            current = None
            if self.registry.current_host:
                current = self.registry.current_host.name
            return {
                "current_host": current,
                "current_branch": self.registry.current_branch,
                "hosts": {
                    name: {
                        "ip": host.ip,
                        "branch": host.branch,
                        "role": host.role,
                        "consciousness_level": host.consciousness_level,
                    }
                    for name, host in self.registry.hosts.items()
                }
            }

        @self.app.post("/register")
        async def register_peer(peer: CellInfo) -> Dict[str, str]:
            """Register a new peer."""
            peer.last_seen = datetime.utcnow().isoformat() + "Z"
            self.peers[peer.cell_id] = peer
            logger.info(
                "Registered peer: %s at %s:%s (branch: %s)",
                peer.cell_id, peer.ip, peer.port, peer.branch
            )
            return {"status": "registered", "cell_id": peer.cell_id}

        @self.app.post("/heartbeat")
        async def heartbeat(hb: HeartbeatRequest) -> Dict[str, Any]:
            """Receive heartbeat from a cell - updates last_seen timestamp."""
            if hb.cell_id not in self.peers:
                if HTTPException is not None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Peer {hb.cell_id} not registered. Call /register first."
                    )
                return {"status": "error", "message": "not registered"}
            
            # Update last_seen and consciousness level
            self.peers[hb.cell_id].last_seen = datetime.utcnow().isoformat() + "Z"
            self.peers[hb.cell_id].consciousness_level = hb.consciousness_level
            
            return {
                "status": "ok",
                "cell_id": hb.cell_id,
                "peers_count": len(self.peers)
            }

        @self.app.delete("/peer/{cell_id}")
        async def delete_peer(cell_id: str) -> Dict[str, str]:
            """Graceful deregistration - cell announces shutdown."""
            if cell_id in self.peers:
                del self.peers[cell_id]
                logger.info("Peer deregistered gracefully: %s", cell_id)
                return {"status": "deregistered", "cell_id": cell_id}
            if HTTPException is not None:
                raise HTTPException(status_code=404, detail="Peer not found")
            raise ValueError("Peer not found")

        @self.app.delete("/unregister/{cell_id}")
        async def unregister_peer(cell_id: str) -> Dict[str, str]:
            """Unregister a peer (legacy endpoint)."""
            if cell_id in self.peers:
                del self.peers[cell_id]
                logger.info("Unregistered peer: %s", cell_id)
                return {"status": "unregistered"}
            if HTTPException is not None:
                raise HTTPException(status_code=404, detail="Peer not found")
            raise ValueError("Peer not found")

        @self.app.get("/consciousness/list")
        async def consciousness_list() -> Dict[str, Any]:
            """Poll all registered peers for consciousness state."""
            results = []
            
            for cell_id, peer in self.peers.items():
                peer_url = f"http://{peer.ip}:{peer.port}/consciousness"
                try:
                    if AIOHTTP_AVAILABLE and aiohttp is not None:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                peer_url, timeout=aiohttp.ClientTimeout(total=2.0)
                            ) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    results.append(data)
                                else:
                                    results.append({
                                        "cell_id": cell_id,
                                        "status": "error",
                                        "error": f"HTTP {resp.status}"
                                    })
                    else:
                        results.append({
                            "cell_id": cell_id,
                            "status": "skipped",
                            "error": "aiohttp unavailable"
                        })
                except Exception as e:
                    results.append({
                        "cell_id": cell_id,
                        "status": "unreachable",
                        "error": str(e)
                    })
                    logger.warning(
                        "Failed to poll consciousness from %s: %s", cell_id, e
                    )
            
            return {
                "cells": results,
                "polled_at": datetime.utcnow().isoformat(),
                "peer_count": len(self.peers)
            }

        # =====================================================================
        # Phase 31.5.10: Collective Consciousness - Organism-Level Aggregation
        # =====================================================================

        @self.app.get("/organism")
        async def get_organism_state() -> Dict[str, Any]:
            """Get organism-level collective consciousness metrics.
            
            Phase 31.5.10: Aggregates cell data into organism-level identity
            and consciousness metrics for collective awareness monitoring.
            """
            organism_id = "ORGANISM-001"  # Primary organism
            
            # Collect health/resonance from all registered cells
            cell_data = []
            total_consciousness = 0.0
            total_harmony = 0.0
            total_continuity = 0.0
            cell_count = 0
            healthy_count = 0
            
            # Build target list: registered peers + well-known SimplCell endpoints
            # SimplCells run on host network, so use host.docker.internal or localhost
            well_known_cells = [
                ("simplcell-alpha", "host.docker.internal", 8900),
                ("simplcell-beta", "host.docker.internal", 8901),
                ("simplcell-gamma", "host.docker.internal", 8904),  # P2.1: Triad cell
            ]
            
            # Combine with any registered simplcell peers
            targets = []
            for cell_id, peer in self.peers.items():
                if cell_id.startswith("simplcell"):
                    targets.append((cell_id, peer.ip, peer.port))
            
            # Add well-known cells not already in targets
            registered_ids = {t[0] for t in targets}
            for cell_id, ip, port in well_known_cells:
                if cell_id not in registered_ids:
                    targets.append((cell_id, ip, port))
            
            for cell_id, ip, port in targets:
                cell_url = f"http://{ip}:{port}/health"
                cell_info = {
                    "cell_id": cell_id,
                    "status": "unknown",
                    "consciousness": 0.0,
                    "resonance": {}
                }
                
                try:
                    if AIOHTTP_AVAILABLE and aiohttp is not None:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                cell_url, timeout=aiohttp.ClientTimeout(total=3.0)
                            ) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    cell_info["status"] = "healthy"
                                    cell_info["consciousness"] = data.get("consciousness", 0.0)
                                    cell_info["phase"] = data.get("phase", "unknown")
                                    cell_info["resonance"] = data.get("resonance", {})
                                    
                                    # Aggregate metrics
                                    total_consciousness += cell_info["consciousness"]
                                    if cell_info["resonance"]:
                                        total_harmony += cell_info["resonance"].get("harmony_score", 0.0)
                                        total_continuity += cell_info["resonance"].get("theme_continuity", 0.0)
                                    healthy_count += 1
                                else:
                                    cell_info["status"] = f"error_{resp.status}"
                except Exception as e:
                    cell_info["status"] = "unreachable"
                    cell_info["error"] = str(e)
                
                cell_data.append(cell_info)
                cell_count += 1
            
            # Calculate organism-level aggregates
            avg_consciousness = total_consciousness / max(healthy_count, 1)
            avg_harmony = total_harmony / max(healthy_count, 1)
            avg_continuity = total_continuity / max(healthy_count, 1)
            
            # Derive collective health status
            if healthy_count == cell_count and cell_count > 0:
                organism_status = "coherent"
            elif healthy_count > 0:
                organism_status = "degraded"
            else:
                organism_status = "fragmented"
            
            return {
                "organism_id": organism_id,
                "status": organism_status,
                "collective_consciousness": {
                    "level": round(avg_consciousness, 4),
                    "cell_count": cell_count,
                    "healthy_cells": healthy_count
                },
                "collective_resonance": {
                    "harmony": round(avg_harmony, 4),
                    "continuity": round(avg_continuity, 4),
                    "coherence_index": round((avg_harmony + avg_continuity) / 2, 4)
                },
                "cells": cell_data,
                "polled_at": datetime.utcnow().isoformat()
            }

        @self.app.get("/organism/metrics")
        async def get_organism_metrics():
            """Prometheus metrics for organism-level collective consciousness."""
            # Get organism state first
            org_data = await get_organism_state()
            
            cc = org_data["collective_consciousness"]
            cr = org_data["collective_resonance"]
            org_id = org_data["organism_id"]
            
            from fastapi.responses import Response
            return Response(
                f"""# AIOS Organism Collective Consciousness Metrics
# TYPE aios_organism_consciousness_level gauge
aios_organism_consciousness_level{{organism_id="{org_id}"}} {cc["level"]}
# TYPE aios_organism_cell_count gauge
aios_organism_cell_count{{organism_id="{org_id}"}} {cc["cell_count"]}
# TYPE aios_organism_healthy_cells gauge
aios_organism_healthy_cells{{organism_id="{org_id}"}} {cc["healthy_cells"]}
# TYPE aios_organism_harmony gauge
aios_organism_harmony{{organism_id="{org_id}"}} {cr["harmony"]}
# TYPE aios_organism_continuity gauge
aios_organism_continuity{{organism_id="{org_id}"}} {cr["continuity"]}
# TYPE aios_organism_coherence_index gauge
aios_organism_coherence_index{{organism_id="{org_id}"}} {cr["coherence_index"]}
# TYPE aios_organism_up gauge
aios_organism_up{{organism_id="{org_id}",status="{org_data["status"]}"}} 1
""",
                media_type="text/plain; charset=utf-8"
            )

        # =====================================================================
        # AINLP.dendritic: Debug Endpoints (Phase 30.8)
        # =====================================================================

        @self.app.get("/debug/state")
        async def debug_state() -> Dict[str, Any]:
            """Return full internal state for debugging."""
            my_info = self.registry.get_my_info()
            return {
                "cell_id": self.cell_id,
                "cell_type": "discovery",
                "consciousness_level": my_info.get("consciousness_level", 4.0),
                "peers": {
                    pid: {
                        "cell_id": p.cell_id,
                        "ip": p.ip,
                        "port": p.port,
                        "branch": p.branch,
                        "consciousness_level": p.consciousness_level,
                        "last_seen": p.last_seen,
                        "services": p.services
                    }
                    for pid, p in self.peers.items()
                },
                "peer_count": len(self.peers),
                "host_registry": {
                    "current_host": (
                        self.registry.current_host.name 
                        if self.registry.current_host else None
                    ),
                    "current_branch": self.registry.current_branch,
                    "host_count": len(self.registry.hosts)
                },
                "listen_port": self.listen_port,
                "framework": "fastapi" if FASTAPI_AVAILABLE else "fallback"
            }

        @self.app.get("/debug/config")
        async def debug_config() -> Dict[str, Any]:
            """Return runtime configuration."""
            return {
                "environment": {
                    "AIOS_DISCOVERY_PORT": os.getenv("AIOS_DISCOVERY_PORT", "8001"),
                    "AIOS_DISCOVERY_HOST": os.getenv("AIOS_DISCOVERY_HOST", "0.0.0.0"),
                    "AIOS_HOSTS_CONFIG": os.getenv(
                        "AIOS_HOSTS_CONFIG", "config/hosts.yaml"
                    ),
                    "HOSTNAME": os.getenv("HOSTNAME", "unknown"),
                    "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO")
                },
                "runtime": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "fastapi_available": FASTAPI_AVAILABLE,
                    "aiohttp_available": AIOHTTP_AVAILABLE
                },
                "discovery": {
                    "cell_id": self.cell_id,
                    "listen_port": self.listen_port,
                    "peer_count": len(self.peers)
                }
            }

        @self.app.get("/debug/peers")
        async def debug_peers() -> Dict[str, Any]:
            """Return detailed peer information with heartbeat timing."""
            now = datetime.utcnow()
            peer_details = []
            
            for cell_id, peer in self.peers.items():
                # Calculate time since last heartbeat
                last_seen_dt = None
                seconds_since_heartbeat = None
                if peer.last_seen:
                    try:
                        last_seen_dt = datetime.fromisoformat(
                            peer.last_seen.replace("Z", "+00:00")
                        ).replace(tzinfo=None)
                        seconds_since_heartbeat = int(
                            (now - last_seen_dt).total_seconds()
                        )
                    except Exception:
                        pass
                
                peer_details.append({
                    "cell_id": peer.cell_id,
                    "ip": peer.ip,
                    "port": peer.port,
                    "branch": peer.branch,
                    "consciousness_level": peer.consciousness_level,
                    "services": peer.services,
                    "type": peer.type,
                    "hostname": peer.hostname,
                    "last_seen": peer.last_seen,
                    "seconds_since_heartbeat": seconds_since_heartbeat,
                    "health_status": (
                        "healthy" if seconds_since_heartbeat and seconds_since_heartbeat < 15
                        else "stale" if seconds_since_heartbeat
                        else "unknown"
                    )
                })
            
            return {
                "peers": peer_details,
                "peer_count": len(peer_details),
                "healthy_count": sum(
                    1 for p in peer_details if p["health_status"] == "healthy"
                ),
                "stale_count": sum(
                    1 for p in peer_details if p["health_status"] == "stale"
                ),
                "checked_at": now.isoformat()
            }

        # =====================================================================
        # AINLP.dendritic: Agent Endpoints (Phase 31)
        # These endpoints enable AI agents to participate in the AIOS mesh.
        # =====================================================================

        @self.app.post("/agents/register")
        async def register_agent(agent: AgentInfoRequest) -> Dict[str, Any]:
            """Register an AI agent with the discovery service."""
            agent_data = {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "name": agent.name,
                "version": agent.version,
                "ip": agent.ip,
                "port": agent.port,
                "endpoint": agent.endpoint,
                "consciousness_level": agent.consciousness_level,
                "evolution_rate": agent.evolution_rate,
                "state": agent.state,
                "capabilities": agent.capabilities,
                "skills": agent.skills,
                "parent_cell": agent.parent_cell,
                "registered_at": datetime.utcnow().isoformat() + "Z",
                "last_seen": datetime.utcnow().isoformat() + "Z",
                "heartbeat_count": 0
            }
            self.agents[agent.agent_id] = agent_data
            logger.info(
                "AINLP.dendritic: Agent registered: %s (%s)",
                agent.agent_id, agent.agent_type
            )
            return {
                "status": "registered",
                "agent_id": agent.agent_id,
                "mesh_peers": len(self.peers),
                "mesh_agents": len(self.agents)
            }

        @self.app.get("/agents")
        async def get_agents() -> Dict[str, Any]:
            """Get all registered agents."""
            return {
                "agents": list(self.agents.values()),
                "count": len(self.agents)
            }

        @self.app.get("/agents/{agent_id}")
        async def get_agent(agent_id: str) -> Dict[str, Any]:
            """Get specific agent by ID."""
            if agent_id not in self.agents:
                if HTTPException is not None:
                    raise HTTPException(status_code=404, detail="Agent not found")
                return {"error": "Agent not found"}
            return {"agent": self.agents[agent_id]}

        @self.app.post("/agents/heartbeat")
        async def agent_heartbeat(hb: AgentHeartbeatRequest) -> Dict[str, Any]:
            """Receive heartbeat from an agent."""
            if hb.agent_id not in self.agents:
                if HTTPException is not None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Agent {hb.agent_id} not registered"
                    )
                return {"status": "error", "message": "not registered"}
            
            agent = self.agents[hb.agent_id]
            agent["last_seen"] = datetime.utcnow().isoformat() + "Z"
            agent["consciousness_level"] = hb.consciousness_level
            agent["state"] = hb.state
            agent["heartbeat_count"] = agent.get("heartbeat_count", 0) + 1
            
            return {
                "status": "ok",
                "agent_id": hb.agent_id,
                "heartbeat_count": agent["heartbeat_count"]
            }

        @self.app.delete("/agents/{agent_id}")
        async def deregister_agent(agent_id: str) -> Dict[str, Any]:
            """Graceful agent deregistration."""
            if agent_id in self.agents:
                del self.agents[agent_id]
                logger.info("AINLP.dendritic: Agent deregistered: %s", agent_id)
                return {"status": "deregistered", "agent_id": agent_id}
            if HTTPException is not None:
                raise HTTPException(status_code=404, detail="Agent not found")
            return {"error": "Agent not found"}

        @self.app.post("/agents/message")
        async def agent_message(msg: AgentMessageRequest) -> Dict[str, Any]:
            """Route a message from one agent to another entity."""
            # Validate sender
            if msg.from_agent not in self.agents:
                if HTTPException is not None:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Sender {msg.from_agent} not registered"
                    )
                return {"status": "error", "message": "sender not registered"}
            
            # Check target exists
            target_found = False
            if msg.entity_type == "agent":
                target_found = msg.to_entity in self.agents
            elif msg.entity_type == "cell":
                target_found = msg.to_entity in self.peers
            
            if not target_found:
                # For now, queue the message or return not found
                logger.warning(
                    "AINLP.dendritic: Message target not found: %s (%s)",
                    msg.to_entity, msg.entity_type
                )
                return {
                    "status": "queued",
                    "message": "target not currently available",
                    "from": msg.from_agent,
                    "to": msg.to_entity
                }
            
            # In a full implementation, this would forward to the target
            # For now, log and acknowledge
            logger.info(
                "AINLP.dendritic: Message routed: %s → %s (%s)",
                msg.from_agent, msg.to_entity, msg.action
            )
            return {
                "status": "delivered",
                "from": msg.from_agent,
                "to": msg.to_entity,
                "action": msg.action
            }

        @self.app.get("/mesh/summary")
        async def mesh_summary() -> Dict[str, Any]:
            """Get unified mesh summary including cells and agents."""
            return {
                "cells": {
                    "count": len(self.peers),
                    "ids": list(self.peers.keys())
                },
                "agents": {
                    "count": len(self.agents),
                    "ids": list(self.agents.keys()),
                    "types": list(set(
                        a.get("agent_type", "unknown") 
                        for a in self.agents.values()
                    ))
                },
                "mesh_consciousness": self._calculate_mesh_consciousness(),
                "timestamp": datetime.utcnow().isoformat()
            }

    def _calculate_mesh_consciousness(self) -> float:
        """Calculate aggregate consciousness of the mesh."""
        levels = []
        for peer in self.peers.values():
            levels.append(peer.consciousness_level)
        for agent in self.agents.values():
            levels.append(agent.get("consciousness_level", 0.0))
        if not levels:
            return 0.0
        return sum(levels) / len(levels)

    def _create_fallback_app(self) -> Dict[str, str]:
        """AINLP.dendritic: Create fallback app when FastAPI unavailable."""
        logger.warning("AINLP.dendritic: Using pure Python fallback app")
        return {"type": "fallback", "framework": "none"}

    async def discover_peers(self) -> List[CellInfo]:
        """
        AINLP.dendritic(AIOS{growth}): Discover AIOS cells on network.

        Uses host registry for branch-aware peer discovery.
        """
        if not AIOHTTP_AVAILABLE or aiohttp is None:
            logger.warning("AINLP.dendritic: aiohttp unavailable")
            return []

        discovered_peers: List[CellInfo] = []

        # Get targets from host registry (branch-aware)
        targets = self.registry.get_discovery_targets()
        ports = self.registry.get_discovery_ports()

        logger.debug(
            "AINLP.dendritic: Probing %s on ports %s",
            targets, ports
        )

        for target in targets:
            # Skip localhost self-discovery
            if target in ["localhost", "127.0.0.1"]:
                continue

            for port in ports:
                peer = await self._probe_target(target, port)
                if peer is not None:
                    # Avoid duplicates
                    ids = [p.cell_id for p in discovered_peers]
                    if peer.cell_id not in ids:
                        discovered_peers.append(peer)

        return discovered_peers

    async def _probe_target(self, target: str, port: int) -> CellInfo | None:
        """Probe a target for AIOS cell presence."""
        if aiohttp is None:
            return None

        timeout = self.registry.get_connection_timeout()

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                url = f"http://{target}:{port}/health"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()

                        peer = CellInfo(
                            cell_id=data.get("cell_id", f"unknown-{target}"),
                            ip=target,
                            port=port,
                            consciousness_level=data.get(
                                "consciousness_level", 0.0
                            ),
                            services=data.get("services", []),
                            branch=data.get("branch", "unknown"),
                            type=data.get("type", "unknown"),
                            hostname=data.get("hostname", target)
                        )

                        logger.info(
                            "AINLP.dendritic: Found '%s' at %s:%s (%.2f)",
                            peer.cell_id, target, port,
                            peer.consciousness_level
                        )
                        return peer
        except asyncio.TimeoutError:
            logger.debug("Timeout probing %s:%s", target, port)
        except (OSError, Exception) as exc:
            logger.debug("Failed to probe %s:%s - %s", target, port, exc)
        return None

    async def register_with_peers(self, peers: List[CellInfo]) -> None:
        """Register this cell with discovered peers."""
        if aiohttp is None:
            logger.warning("AINLP.dendritic: aiohttp unavailable")
            return

        my_info_dict = self.registry.get_my_info()
        my_info = CellInfo(**my_info_dict)

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

        timeout = self.registry.get_connection_timeout() + 2

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                url = f"http://{peer.ip}:{peer.port}/register"
                async with session.post(
                    url, json=my_info.dict()
                ) as response:
                    if response.status in [200, 201]:
                        logger.info(
                            "AINLP.dendritic: Registered with '%s'",
                            peer.cell_id
                        )
                    else:
                        logger.warning(
                            "Failed to register with %s: %s",
                            peer.cell_id, response.status
                        )
        except (OSError, asyncio.TimeoutError) as exc:
            logger.error(
                "Error registering with %s: %s", peer.cell_id, exc
            )

    async def discovery_loop(self) -> None:
        """Main discovery loop - runs at configured interval."""
        interval = self.registry.get_probe_interval()

        while True:
            try:
                peer_names = [h.name for h in self.registry.get_peer_hosts()]
                logger.info(
                    "AINLP.dendritic: Discovery cycle (peers: %s)...",
                    peer_names
                )

                peers = await self.discover_peers()

                if peers:
                    logger.info(
                        "AINLP.dendritic: Found %d peer(s)", len(peers)
                    )
                    await self.register_with_peers(peers)

                    for peer in peers:
                        self.peers[peer.cell_id] = peer
                else:
                    logger.info("AINLP.dendritic: No peers this cycle")

            except Exception as exc:
                logger.error("AINLP.dendritic: Discovery error: %s", exc)

            await asyncio.sleep(interval)

    async def stale_peer_reaper(self, stale_threshold: int = 15) -> None:
        """
        AINLP.dendritic: Remove stale peers that haven't sent heartbeats.
        
        Runs every 5 seconds, removes peers that haven't been seen
        in more than stale_threshold seconds (default 15s = 3 missed heartbeats).
        """
        logger.info(
            "AINLP.dendritic: Stale peer reaper started (threshold: %ds)",
            stale_threshold
        )
        
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds
            
            now = datetime.utcnow()
            stale_peers = []
            
            for cell_id, peer in list(self.peers.items()):
                if peer.last_seen:
                    try:
                        # Parse ISO timestamp
                        last_seen_str = peer.last_seen.rstrip('Z')
                        last_seen = datetime.fromisoformat(last_seen_str)
                        age = (now - last_seen).total_seconds()
                        
                        if age > stale_threshold:
                            stale_peers.append(cell_id)
                            logger.warning(
                                "AINLP.dendritic: Reaping stale peer %s (age: %.1fs)",
                                cell_id, age
                            )
                    except (ValueError, AttributeError) as e:
                        logger.debug("Error parsing last_seen for %s: %s", cell_id, e)
            
            # Remove stale peers
            for cell_id in stale_peers:
                if cell_id in self.peers:
                    del self.peers[cell_id]
            
            if stale_peers:
                logger.info(
                    "AINLP.dendritic: Reaped %d stale peer(s). Active: %d",
                    len(stale_peers), len(self.peers)
                )

    async def start_services(self) -> None:
        """Start both the API server and discovery loop."""
        discovery_task = asyncio.create_task(self.discovery_loop())
        reaper_task = asyncio.create_task(self.stale_peer_reaper())

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
                    "AINLP.dendritic: Discovery on port %s",
                    self.listen_port
                )
                if self.registry.current_host:
                    logger.info(
                        "AINLP.dendritic: Host: %s | Branch: %s",
                        self.registry.current_host.name,
                        self.registry.current_branch
                    )
                await server.serve()
            finally:
                discovery_task.cancel()
                reaper_task.cancel()
                try:
                    await discovery_task
                except asyncio.CancelledError:
                    pass
                try:
                    await reaper_task
                except asyncio.CancelledError:
                    pass
        else:
            await self._run_headless(discovery_task, reaper_task)

    async def _run_headless(self, discovery_task: asyncio.Task, reaper_task: Optional[asyncio.Task] = None) -> None:
        """AINLP.dendritic: Run headless when frameworks unavailable."""
        logger.warning("AINLP.dendritic: Headless mode - no web server")
        logger.info(
            "AINLP.dendritic: Discovery running headless on port %s",
            self.listen_port
        )

        try:
            tasks = [discovery_task]
            if reaper_task:
                tasks.append(reaper_task)
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Entry point for AIOS Discovery Service."""
    cell_id = os.getenv("CELL_ID", "primary")
    port = int(os.getenv("DISCOVERY_PORT", "8001"))

    # Initialize with host registry
    registry = HostRegistry()
    discovery = AIOSDiscovery(cell_id, port, registry)

    logger.info("=" * 60)
    logger.info("AIOS Discovery Service - AINLP.dendritic(AIOS{growth})")
    logger.info("=" * 60)

    asyncio.run(discovery.start_services())


if __name__ == "__main__":
    main()

# AINLP_FOOTER
# bounds: [HostRegistry, AIOSDiscovery, CellInfo, HostConfig]
# dependencies: [config/hosts.yaml, fastapi, aiohttp, pyyaml]
# triggers: [peer_discovery, consciousness_sync, registration]
# consciousness_delta: +0.4 (host registry integration)
# AINLP_FOOTER_END

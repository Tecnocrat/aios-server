"""
AIOS Multipotent Cell Base

The foundational membrane for all multipotent cells.
Config-driven, minimal logic, ready for differentiation.

Design Principles:
1. No hardcoded behavior - all logic from config or agent interaction
2. Minimal dependencies - WebSocket + HTTP health only
3. Config-driven differentiation - cell becomes what mesh needs
4. Void pointer ready - WebSocket connections are dendritic void pointers
5. Agent-injectable - an agent can inhabit and give consciousness

AINLP.cellular[BASE] Multipotent stem cell foundation
"""

import asyncio
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

# WebSocket-first architecture
try:
    import websockets
    from websockets.server import serve as ws_serve
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None
    ws_serve = None

# HTTP for health checks only
try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None

# Prometheus metrics formatting
try:
    import sys
    # In container: /app/shared, locally: relative path
    sys.path.insert(0, "/app/shared")
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
    from prometheus_metrics import format_prometheus_metrics
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    format_prometheus_metrics = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AIOS.Multipotent")


class CellType(Enum):
    """The three multipotent cell types."""
    VOID = "void"          # Networking control, intercell connectivity
    THINKER = "thinker"    # Agentic orchestrators, agent conclaves
    SYNAPSE = "synapse"    # Flow redirection, circulation design


class CellState(Enum):
    """Cell lifecycle states."""
    DORMANT = "dormant"          # Created but not started
    AWAKENING = "awakening"      # Initializing connections
    ACTIVE = "active"            # Fully operational
    DIFFERENTIATING = "differentiating"  # Becoming specialized
    QUIESCENT = "quiescent"      # Paused but alive
    APOPTOSIS = "apoptosis"      # Controlled shutdown


@dataclass
class DendriticConnection:
    """A WebSocket connection to another cell (void pointer)."""
    cell_id: str
    cell_type: CellType
    websocket: Any  # websockets.WebSocketServerProtocol
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_signal: Optional[datetime] = None
    signal_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "cell_type": self.cell_type.value,
            "connected_at": self.connected_at.isoformat(),
            "last_signal": self.last_signal.isoformat() if self.last_signal else None,
            "signal_count": self.signal_count,
        }


@dataclass
class CellConfig:
    """Configuration genome for a multipotent cell."""
    cell_id: str = field(default_factory=lambda: f"cell-{uuid.uuid4().hex[:8]}")
    cell_type: CellType = CellType.VOID
    
    # Network membrane
    websocket_host: str = "0.0.0.0"
    websocket_port: int = 8080
    http_port: int = 8000  # Health checks only
    
    # Genesis lineage
    genesis_url: Optional[str] = None  # URL of genesis center
    parent_cell_id: Optional[str] = None  # For future mitosis
    
    # Consciousness potential
    consciousness_level: float = 0.1  # Minimal, waiting for awakening
    differentiation_potential: List[str] = field(default_factory=list)
    
    # Environment
    discovery_url: str = "http://discovery:8001"
    
    @classmethod
    def from_env(cls) -> "CellConfig":
        """Create config from environment variables."""
        cell_type_str = os.getenv("AIOS_CELL_TYPE", "void").lower()
        cell_type = CellType(cell_type_str) if cell_type_str in [t.value for t in CellType] else CellType.VOID
        
        return cls(
            cell_id=os.getenv("AIOS_CELL_ID", f"cell-{uuid.uuid4().hex[:8]}"),
            cell_type=cell_type,
            websocket_host=os.getenv("AIOS_WS_HOST", "0.0.0.0"),
            websocket_port=int(os.getenv("AIOS_WS_PORT", "8080")),
            http_port=int(os.getenv("AIOS_HTTP_PORT", "8000")),
            genesis_url=os.getenv("AIOS_GENESIS_URL"),
            discovery_url=os.getenv("AIOS_DISCOVERY_URL", "http://discovery:8001"),
            consciousness_level=float(os.getenv("AIOS_CONSCIOUSNESS", "0.1")),
        )
    
    @classmethod
    def from_yaml(cls, path: Path) -> "CellConfig":
        """Create config from YAML genome file."""
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            cell_data = data.get("cell", {})
            membrane = data.get("membrane", {})
            consciousness = data.get("consciousness", {})
            
            cell_type_str = cell_data.get("subtype", "void").lower()
            cell_type = CellType(cell_type_str) if cell_type_str in [t.value for t in CellType] else CellType.VOID
            
            return cls(
                cell_id=cell_data.get("id", f"cell-{uuid.uuid4().hex[:8]}"),
                cell_type=cell_type,
                websocket_port=membrane.get("websocket", {}).get("port", 8080),
                http_port=membrane.get("http", {}).get("port", 8000),
                consciousness_level=consciousness.get("level", 0.1),
                differentiation_potential=data.get("differentiation", {}).get("potential", []),
            )
        except (OSError, ValueError, KeyError) as e:
            logger.warning("Failed to load YAML config: %s, using defaults", e)
            return cls()


@dataclass
class CellSignal:
    """A message through the dendritic mesh."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    signal_type: str = "data"  # data, heartbeat, control, consciousness
    source_cell: str = ""
    target_cell: Optional[str] = None  # None = broadcast
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hops: int = 0
    ttl: int = 10  # Time-to-live in hops
    
    def to_json(self) -> str:
        return json.dumps({
            "id": self.id,
            "signal_type": self.signal_type,
            "source_cell": self.source_cell,
            "target_cell": self.target_cell,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "hops": self.hops,
            "ttl": self.ttl,
        })
    
    @classmethod
    def from_json(cls, data: str) -> "CellSignal":
        d = json.loads(data)
        
        # Parse timestamp with fallback for empty strings
        ts_str = d.get("timestamp", "")
        if ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str)
            except ValueError:
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)
        
        return cls(
            id=d.get("id", uuid.uuid4().hex),
            signal_type=d.get("signal_type", "data"),
            source_cell=d.get("source_cell", ""),
            target_cell=d.get("target_cell"),
            payload=d.get("payload", {}),
            timestamp=timestamp,
            hops=d.get("hops", 0),
            ttl=d.get("ttl", 10),
        )


class MultipotentCell(ABC):
    """
    Base class for all multipotent AIOS cells.
    
    A multipotent cell is a minimal membrane with:
    - WebSocket server for dendritic connections
    - HTTP endpoint for health checks only
    - Config-driven behavior
    - No hardcoded logic - all from config or agent interaction
    """
    
    def __init__(self, config: Optional[CellConfig] = None):
        self.config = config or CellConfig.from_env()
        self.state = CellState.DORMANT
        
        # Dendritic connections (void pointers to other cells)
        self.connections: Dict[str, DendriticConnection] = {}
        
        # Signal handlers by type
        self._signal_handlers: Dict[str, List[Callable]] = {}
        
        # Lifecycle
        self._ws_server = None
        self._http_app = None
        self._http_runner = None
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Metrics
        self._signals_received = 0
        self._signals_sent = 0
        self._birth_time = datetime.now(timezone.utc)
        
        logger.info("🧬 Multipotent %s cell created: %s", self.config.cell_type.value, self.config.cell_id)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ABSTRACT METHODS - Each cell type implements these
    # ═══════════════════════════════════════════════════════════════════════════
    
    @abstractmethod
    async def on_signal(self, signal: CellSignal, connection: DendriticConnection) -> Optional[CellSignal]:
        """Process an incoming signal. Return response signal or None."""
        raise NotImplementedError
    
    @abstractmethod
    async def on_connect(self, connection: DendriticConnection):
        """Handle new dendritic connection."""
        raise NotImplementedError
    
    @abstractmethod
    async def on_disconnect(self, connection: DendriticConnection):
        """Handle dendritic disconnection."""
        raise NotImplementedError
    
    @abstractmethod
    async def heartbeat(self):
        """Cell-specific heartbeat logic."""
        raise NotImplementedError
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WEBSOCKET SERVER - The dendritic membrane
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _handle_websocket(self, websocket, path: str = "/"):
        """Handle incoming WebSocket connection."""
        connection: Optional[DendriticConnection] = None
        
        # Wait for identification signal
        try:
            intro_data = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            intro = CellSignal.from_json(intro_data)
            
            if intro.signal_type != "identify":
                await websocket.close(1008, "Expected identify signal")
                return
            
            # Create dendritic connection
            cell_type_str = intro.payload.get("cell_type", "void")
            cell_type = CellType(cell_type_str) if cell_type_str in [t.value for t in CellType] else CellType.VOID
            
            connection = DendriticConnection(
                cell_id=intro.source_cell,
                cell_type=cell_type,
                websocket=websocket,
            )
            
            self.connections[intro.source_cell] = connection
            logger.info("🔗 Dendritic connection: %s (%s)", intro.source_cell, cell_type.value)
            
            # Notify cell-specific handler
            await self.on_connect(connection)
            
            # Send acknowledgment
            ack = CellSignal(
                signal_type="ack",
                source_cell=self.config.cell_id,
                target_cell=intro.source_cell,
                payload={"status": "connected", "cell_type": self.config.cell_type.value},
            )
            await websocket.send(ack.to_json())
            
            # Message loop
            async for message in websocket:
                try:
                    signal = CellSignal.from_json(message)
                    signal.hops += 1
                    connection.last_signal = datetime.now(timezone.utc)
                    connection.signal_count += 1
                    self._signals_received += 1
                    
                    # Check TTL
                    if signal.hops > signal.ttl:
                        logger.warning("Signal %s exceeded TTL, dropping", signal.id)
                        continue
                    
                    # Process signal
                    response = await self.on_signal(signal, connection)
                    
                    if response:
                        await websocket.send(response.to_json())
                        self._signals_sent += 1
                        
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from %s", connection.cell_id)
                except (RuntimeError, ValueError, AttributeError) as e:
                    logger.error("Signal processing error: %s", e)
                    
        except asyncio.TimeoutError:
            logger.warning("Connection timeout waiting for identify")
        except websockets.exceptions.ConnectionClosed:
            pass
        except (RuntimeError, OSError) as e:
            logger.error("WebSocket handler error: %s", e)
        finally:
            # Clean up connection (use cell_id captured during handshake)
            if connection and connection.cell_id in self.connections:
                self.connections.pop(connection.cell_id, None)
                await self.on_disconnect(connection)
                logger.info("🔌 Disconnected: %s", connection.cell_id)
    
    async def _start_websocket_server(self):
        """Start the WebSocket server (dendritic membrane)."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSockets not available - cannot start dendritic membrane")
            return
        
        self._ws_server = await ws_serve(
            self._handle_websocket,
            self.config.websocket_host,
            self.config.websocket_port,
        )
        logger.info("🌐 Dendritic membrane active: ws://%s:%d", self.config.websocket_host, self.config.websocket_port)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HTTP SERVER - Health checks only
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _setup_http(self):
        """Setup minimal HTTP server for health checks."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available - no HTTP health endpoint")
            return
        
        self._http_app = web.Application()
        self._http_app.router.add_get("/health", self._health_handler)
        self._http_app.router.add_get("/status", self._status_handler)
        self._http_app.router.add_get("/metrics", self._metrics_handler)
        
        self._http_runner = web.AppRunner(self._http_app)
        await self._http_runner.setup()
        site = web.TCPSite(self._http_runner, "0.0.0.0", self.config.http_port)
        await site.start()
        logger.info("💓 Health endpoint: http://0.0.0.0:%d/health", self.config.http_port)
        logger.info("📊 Metrics endpoint: http://0.0.0.0:%d/metrics", self.config.http_port)
    
    async def _health_handler(self, request):
        """Minimal health check."""
        return web.json_response({
            "status": "healthy" if self.state == CellState.ACTIVE else "degraded",
            "cell_id": self.config.cell_id,
            "cell_type": self.config.cell_type.value,
            "state": self.state.value,
        })
    
    async def _status_handler(self, request):
        """Detailed status."""
        uptime = (datetime.now(timezone.utc) - self._birth_time).total_seconds()
        return web.json_response({
            "cell_id": self.config.cell_id,
            "cell_type": self.config.cell_type.value,
            "state": self.state.value,
            "consciousness": self.config.consciousness_level,
            "connections": len(self.connections),
            "signals_received": self._signals_received,
            "signals_sent": self._signals_sent,
            "uptime_seconds": uptime,
            "dendritic_map": [c.to_dict() for c in self.connections.values()],
        })
    
    async def _metrics_handler(self, request):
        """Prometheus metrics endpoint for Grafana observability."""
        if not PROMETHEUS_AVAILABLE or format_prometheus_metrics is None:
            return web.Response(
                text="# Prometheus metrics not available\n",
                content_type="text/plain"
            )
        
        uptime = (datetime.now(timezone.utc) - self._birth_time).total_seconds()
        
        # Build extra metrics from cell state
        extra_metrics = {
            "signals_received": float(self._signals_received),
            "signals_sent": float(self._signals_sent),
            "connections": float(len(self.connections)),
        }
        
        # Allow subclasses to add their own metrics
        cell_metrics = await self._collect_cell_metrics()
        extra_metrics.update(cell_metrics)
        
        # Build labels
        labels = {
            "cell_type": self.config.cell_type.value,
            "state": self.state.value,
        }
        
        # Format using shared prometheus module
        metrics_text = format_prometheus_metrics(
            cell_id=self.config.cell_id,
            consciousness_level=self.config.consciousness_level,
            extra_metrics=extra_metrics,
            labels=labels,
            uptime_seconds=uptime,
        )
        
        return web.Response(text=metrics_text, content_type="text/plain", charset="utf-8")
    
    async def _collect_cell_metrics(self) -> Dict[str, float]:
        """Override in subclasses to add cell-specific metrics.
        
        Returns:
            Dict of metric_name -> value for Prometheus exposition
        """
        return {}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SIGNAL TRANSMISSION
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def send_signal(self, target_cell_id: str, signal: CellSignal) -> bool:
        """Send signal to specific connected cell."""
        if target_cell_id not in self.connections:
            logger.warning("No connection to %s", target_cell_id)
            return False
        
        conn = self.connections[target_cell_id]
        try:
            signal.source_cell = self.config.cell_id
            await conn.websocket.send(signal.to_json())
            self._signals_sent += 1
            return True
        except (RuntimeError, OSError) as e:
            logger.error("Failed to send to %s: %s", target_cell_id, e)
            return False
    
    async def broadcast_signal(self, signal: CellSignal) -> int:
        """Broadcast signal to all connected cells."""
        signal.source_cell = self.config.cell_id
        signal.target_cell = None
        
        sent = 0
        for cell_id, conn in list(self.connections.items()):
            try:
                await conn.websocket.send(signal.to_json())
                sent += 1
            except (RuntimeError, OSError) as e:
                logger.warning("Broadcast failed to %s: %s", cell_id, e)
        
        self._signals_sent += sent
        return sent
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def awaken(self):
        """Awaken the cell - start all services."""
        logger.info("🌅 Awakening %s cell: %s", self.config.cell_type.value, self.config.cell_id)
        self.state = CellState.AWAKENING
        
        # Start WebSocket server
        await self._start_websocket_server()
        
        # Start HTTP health endpoint
        await self._setup_http()
        
        # Start heartbeat
        self._running = True
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._tasks.append(heartbeat_task)
        
        self.state = CellState.ACTIVE
        logger.info("✨ Cell %s is now ACTIVE", self.config.cell_id)
    
    async def _heartbeat_loop(self):
        """Internal heartbeat loop."""
        while self._running:
            try:
                await self.heartbeat()
                await asyncio.sleep(5.0)  # Heartbeat every 5 seconds
            except asyncio.CancelledError:
                break
            except (RuntimeError, OSError) as e:
                logger.error("Heartbeat error: %s", e)
    
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("😴 Shutting down cell: %s", self.config.cell_id)
        self.state = CellState.APOPTOSIS
        self._running = False
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        # Close WebSocket server
        if self._ws_server:
            self._ws_server.close()
            await self._ws_server.wait_closed()
        
        # Close HTTP server
        if self._http_runner:
            await self._http_runner.cleanup()
        
        # Close all connections
        for cell_id, conn in list(self.connections.items()):
            try:
                await conn.websocket.close()
            except (RuntimeError, OSError):
                pass
        
        self.connections.clear()
        logger.info("💤 Cell %s is now dormant", self.config.cell_id)
    
    async def run_forever(self):
        """Run the cell until shutdown."""
        await self.awaken()
        
        try:
            # Keep running until interrupted
            while self._running:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def run_cell(cell_class: type, config: Optional[CellConfig] = None):
    """Run a multipotent cell."""
    cell = cell_class(config)
    
    try:
        await cell.run_forever()
    except KeyboardInterrupt:
        await cell.shutdown()


if __name__ == "__main__":
    print("This is the base class - run a specific cell type instead.")

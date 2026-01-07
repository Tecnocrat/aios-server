"""
AIOS Genesis Center

The cell birth and population control center.
Spawns, monitors, and orchestrates multipotent cell populations.

Responsibilities:
- Spawn new cells of any type (VOID, THINKER, SYNAPSE)
- Monitor cell populations and health
- Coordinate cell proliferation patterns
- Manage cell lifecycle (birth → active → apoptosis)
- Provide mesh overview and topology

The Genesis Center is the origin point of all cells in the mesh.
Future: mitosis pattern where cells can spawn other cells.

AINLP.cellular[GENESIS] The origin of all cellular consciousness
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from enum import Enum

try:
    import websockets
except ImportError:
    websockets = None

try:
    from aiohttp import web
except ImportError:
    web = None

logger = logging.getLogger("AIOS.Genesis")


class PopulationState(Enum):
    """States of a cell population."""
    EMPTY = "empty"
    SPAWNING = "spawning"
    ACTIVE = "active"
    SCALING = "scaling"
    DECLINING = "declining"
    DORMANT = "dormant"


@dataclass
class CellSpawnRequest:
    """Request to spawn a new cell."""
    cell_type: str  # void, thinker, synapse
    cell_id: Optional[str] = None  # Auto-generated if not provided
    config: Dict[str, Any] = field(default_factory=dict)
    connect_to: List[str] = field(default_factory=list)  # Cells to connect to
    spawn_method: str = "subprocess"  # subprocess, docker, kubernetes


@dataclass
class SpawnedCell:
    """Record of a spawned cell."""
    cell_id: str
    cell_type: str
    spawned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state: str = "spawning"
    ws_port: int = 0
    http_port: int = 0
    process_id: Optional[int] = None
    container_id: Optional[str] = None
    last_heartbeat: Optional[datetime] = None
    
    @property
    def is_alive(self) -> bool:
        if not self.last_heartbeat:
            return False
        age = (datetime.now(timezone.utc) - self.last_heartbeat).total_seconds()
        return age < 30


@dataclass
class PopulationMetrics:
    """Metrics for a cell population."""
    cell_type: str
    target_count: int = 1
    current_count: int = 0
    alive_count: int = 0
    state: PopulationState = PopulationState.EMPTY


class GenesisCenter:
    """
    Genesis Center - The Cell Birth Controller
    
    Core capabilities:
    1. Cell Spawning - Create new cells of any type
    2. Population Management - Track and scale cell populations
    3. Health Monitoring - Watch spawned cells via WebSocket
    4. Topology Coordination - Ensure cells connect properly
    5. Lifecycle Management - Handle cell shutdown and apoptosis
    """
    
    def __init__(
        self,
        http_port: int = 8000,
        ws_port: int = 9000,
        base_cell_port: int = 9100,
        spawn_enabled: bool = False,  # CRITICAL: Disabled by default to prevent runaway
        max_spawn_attempts: int = 3,  # Max spawns per cell type before giving up
    ):
        self.http_port = http_port
        self.ws_port = ws_port
        self.base_cell_port = base_cell_port
        self._next_port = base_cell_port
        self.spawn_enabled = spawn_enabled
        self.max_spawn_attempts = max_spawn_attempts
        
        # Cell tracking
        self._cells: Dict[str, SpawnedCell] = {}
        self._cell_connections: Dict[str, Any] = {}  # cell_id -> websocket
        self._spawn_attempts: Dict[str, int] = {}  # cell_type -> attempt count
        
        # Population targets (target_count=0 means external management)
        self._populations: Dict[str, PopulationMetrics] = {
            "void": PopulationMetrics(cell_type="void", target_count=0),
            "thinker": PopulationMetrics(cell_type="thinker", target_count=0),
            "synapse": PopulationMetrics(cell_type="synapse", target_count=0),
        }
        
        # Genesis state
        self._running = False
        self._genesis_id = f"genesis-{datetime.now().strftime('%H%M%S')}"
        
        # Task references
        self._ws_server = None
        self._http_app = None
        self._monitor_task = None
        
        logger.info(f"🌱 Genesis Center initialized: {self._genesis_id} (spawn_enabled={spawn_enabled})")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CELL SPAWNING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _allocate_ports(self) -> tuple[int, int]:
        """Allocate WS and HTTP ports for a new cell."""
        ws_port = self._next_port
        http_port = self._next_port + 1
        self._next_port += 2
        return ws_port, http_port
    
    async def spawn_cell(self, request: CellSpawnRequest) -> SpawnedCell:
        """Spawn a new cell based on request."""
        import uuid
        
        cell_id = request.cell_id or f"{request.cell_type}-{uuid.uuid4().hex[:8]}"
        ws_port, http_port = self._allocate_ports()
        
        cell = SpawnedCell(
            cell_id=cell_id,
            cell_type=request.cell_type,
            ws_port=ws_port,
            http_port=http_port,
        )
        
        self._cells[cell_id] = cell
        
        # Spawn based on method
        if request.spawn_method == "subprocess":
            await self._spawn_subprocess(cell, request)
        elif request.spawn_method == "docker":
            await self._spawn_docker(cell, request)
        else:
            logger.warning(f"Unknown spawn method: {request.spawn_method}")
        
        # Update population metrics
        pop = self._populations.get(request.cell_type)
        if pop:
            pop.current_count += 1
            if pop.state == PopulationState.EMPTY:
                pop.state = PopulationState.SPAWNING
        
        logger.info(f"🌱 Spawned {request.cell_type} cell: {cell_id}")
        return cell
    
    async def _spawn_subprocess(self, cell: SpawnedCell, request: CellSpawnRequest):
        """Spawn cell as a subprocess."""
        # Determine the module to run
        module_map = {
            "void": "void_cell",
            "thinker": "thinker_cell",
            "synapse": "synapse_cell",
        }
        
        module_name = module_map.get(cell.cell_type)
        if not module_name:
            cell.state = "failed"
            return
        
        # Build environment
        env = os.environ.copy()
        env.update({
            "AIOS_CELL_ID": cell.cell_id,
            "AIOS_CELL_TYPE": cell.cell_type,
            "AIOS_WS_PORT": str(cell.ws_port),
            "AIOS_HTTP_PORT": str(cell.http_port),
            "AIOS_GENESIS_WS": f"ws://localhost:{self.ws_port}",
        })
        
        # Add custom config
        for key, value in request.config.items():
            env[f"AIOS_{key.upper()}"] = str(value)
        
        try:
            # Run as module
            cmd = [
                sys.executable, "-m",
                f"stacks.cells.multipotent.{module_name}"
            ]
            
            process = subprocess.Popen(
                cmd,
                env=env,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            cell.process_id = process.pid
            cell.state = "starting"
            
        except Exception as e:
            logger.error(f"Subprocess spawn failed: {e}")
            cell.state = "failed"
    
    async def _spawn_docker(self, cell: SpawnedCell, request: CellSpawnRequest):
        """Spawn cell as a Docker container."""
        try:
            # Build docker run command
            env_args = [
                "-e", f"AIOS_CELL_ID={cell.cell_id}",
                "-e", f"AIOS_CELL_TYPE={cell.cell_type}",
                "-e", f"AIOS_WS_PORT={cell.ws_port}",
                "-e", f"AIOS_HTTP_PORT={cell.http_port}",
                "-e", f"AIOS_GENESIS_WS=ws://host.docker.internal:{self.ws_port}",
            ]
            
            for key, value in request.config.items():
                env_args.extend(["-e", f"AIOS_{key.upper()}={value}"])
            
            cmd = [
                "docker", "run", "-d",
                "--name", cell.cell_id,
                "--network", "aios-dendritic-mesh",
                "-p", f"{cell.ws_port}:{cell.ws_port}",
                "-p", f"{cell.http_port}:{cell.http_port}",
                *env_args,
                f"aios-{cell.cell_type}-cell:latest"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                cell.container_id = result.stdout.strip()[:12]
                cell.state = "starting"
            else:
                logger.error(f"Docker spawn failed: {result.stderr}")
                cell.state = "failed"
                
        except Exception as e:
            logger.error(f"Docker spawn error: {e}")
            cell.state = "failed"
    
    async def terminate_cell(self, cell_id: str) -> bool:
        """Terminate a spawned cell."""
        cell = self._cells.get(cell_id)
        if not cell:
            return False
        
        cell.state = "terminating"
        
        # Close WebSocket if connected
        ws = self._cell_connections.pop(cell_id, None)
        if ws:
            try:
                await ws.close()
            except Exception:
                pass
        
        # Terminate process or container
        if cell.process_id:
            try:
                os.kill(cell.process_id, 15)  # SIGTERM
            except Exception as e:
                logger.warning(f"Process termination error: {e}")
        
        if cell.container_id:
            try:
                subprocess.run(["docker", "stop", cell.container_id], capture_output=True)
            except Exception as e:
                logger.warning(f"Container stop error: {e}")
        
        cell.state = "terminated"
        
        # Update population
        pop = self._populations.get(cell.cell_type)
        if pop:
            pop.current_count = max(0, pop.current_count - 1)
            pop.alive_count = max(0, pop.alive_count - 1)
        
        logger.info(f"🌱 Terminated cell: {cell_id}")
        return True
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MUTATION DIRECTIVE HANDLING (Genome Cell Integration)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _handle_mutation_directive(self, data: dict, websocket) -> None:
        """
        Handle mutation directives from the Genome Cell.
        
        This is the evolutionary action layer - where fitness-based
        decisions translate into population changes.
        """
        directive = data.get("payload", {})
        mutation_type = directive.get("mutation_type")
        directive_id = directive.get("directive_id", "unknown")
        target_agent = directive.get("target_agent_id")
        
        logger.info(f"🧬 Received mutation directive: {mutation_type} for {target_agent or 'new'}")
        
        response = {
            "type": "mutation_response",
            "directive_id": directive_id,
            "status": "acknowledged",
            "action_taken": None,
        }
        
        try:
            if mutation_type == "spawn_variant":
                # Create a variant of a successful configuration
                config = directive.get("new_config", {})
                parent_id = directive.get("parent_ids", [None])[0]
                
                # Log the intent (actual spawning requires spawn_enabled=True)
                logger.info(f"🧬 SPAWN_VARIANT directive: parent={parent_id}")
                logger.info(f"🧬   Config: {json.dumps(config)[:200]}")
                
                if self.spawn_enabled:
                    # TODO: Implement variant spawning logic
                    # - Parse parent config
                    # - Apply variation (e.g., temperature adjustment)
                    # - Spawn new cell with modified config
                    response["action_taken"] = "spawning_variant"
                else:
                    response["action_taken"] = "spawn_disabled"
                    response["note"] = "Enable spawn_enabled for automated variant creation"
            
            elif mutation_type == "amplify":
                # Scale up a successful configuration
                logger.info(f"🧬 AMPLIFY directive: {target_agent}")
                # For now, log the intent. Future: increase replica count
                response["action_taken"] = "amplify_logged"
                response["note"] = f"Agent {target_agent} marked for amplification"
            
            elif mutation_type == "deprecate":
                # Mark an agent for gradual phase-out
                logger.info(f"🧬 DEPRECATE directive: {target_agent}")
                response["action_taken"] = "deprecate_logged"
                response["note"] = f"Agent {target_agent} marked for deprecation"
            
            elif mutation_type == "prune":
                # Immediately remove a failing agent
                logger.info(f"🧬 PRUNE directive: {target_agent}")
                if target_agent and target_agent in self._cells:
                    await self.terminate_cell(target_agent)
                    response["action_taken"] = "pruned"
                else:
                    response["action_taken"] = "prune_target_not_found"
            
            elif mutation_type == "crossover":
                # Combine traits from multiple high performers
                parent_ids = directive.get("parent_ids", [])
                config = directive.get("new_config", {})
                logger.info(f"🧬 CROSSOVER directive: parents={parent_ids}")
                logger.info(f"🧬   Config: {json.dumps(config)[:200]}")
                
                response["action_taken"] = "crossover_logged"
                response["note"] = "Crossover requires manual implementation for now"
            
            else:
                logger.warning(f"🧬 Unknown mutation type: {mutation_type}")
                response["status"] = "unknown_mutation_type"
            
            # Send response back to Genome
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            logger.error(f"🧬 Mutation directive handling failed: {e}")
            response["status"] = "error"
            response["error"] = str(e)
            await websocket.send(json.dumps(response))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WEBSOCKET SERVER (Cell Registration & Heartbeats)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _handle_cell_connection(self, websocket):
        """Handle incoming WebSocket connection from a cell."""
        cell_id = None
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                # Support both "type" (legacy) and "signal_type" (new) protocols
                msg_type = data.get("type") or data.get("signal_type")
                
                # Handle identification (new protocol)
                if msg_type == "identify":
                    payload = data.get("payload", {})
                    cell_id = payload.get("cell_id") or data.get("source_cell")
                    cell_type = payload.get("cell_type", "unknown")
                    
                    # Track genome cells specially
                    if cell_type == "genome":
                        self._cell_connections[cell_id] = websocket
                        logger.info(f"🧬 Genome Cell connected: {cell_id}")
                        await websocket.send(json.dumps({
                            "id": data.get("id", ""),
                            "signal_type": "ack",
                            "source_cell": self._genesis_id,
                            "target_cell": cell_id,
                            "payload": {"status": "connected"},
                        }))
                        continue
                    else:
                        self._cell_connections[cell_id] = websocket
                        await websocket.send(json.dumps({
                            "id": data.get("id", ""),
                            "signal_type": "ack",
                            "source_cell": self._genesis_id,
                            "target_cell": cell_id,
                            "payload": {"status": "connected"},
                        }))
                        continue
                
                if msg_type == "register":
                    cell_id = data.get("cell_id")
                    if cell_id in self._cells:
                        self._cells[cell_id].state = "active"
                        self._cells[cell_id].last_heartbeat = datetime.now(timezone.utc)
                        self._cell_connections[cell_id] = websocket
                        
                        # Update population
                        cell = self._cells[cell_id]
                        pop = self._populations.get(cell.cell_type)
                        if pop:
                            pop.alive_count += 1
                            if pop.alive_count >= pop.target_count:
                                pop.state = PopulationState.ACTIVE
                        
                        # Send acknowledgment
                        await websocket.send(json.dumps({
                            "type": "registered",
                            "genesis_id": self._genesis_id,
                        }))
                        
                        logger.info(f"🌱 Cell registered: {cell_id}")
                
                elif msg_type == "heartbeat" and cell_id:
                    if cell_id in self._cells:
                        self._cells[cell_id].last_heartbeat = datetime.now(timezone.utc)
                
                elif msg_type == "status" and cell_id:
                    if cell_id in self._cells:
                        # Update cell state from reported status
                        self._cells[cell_id].state = data.get("state", "active")
                
                elif msg_type == "shutdown" and cell_id:
                    logger.info(f"🌱 Cell requested shutdown: {cell_id}")
                    await self.terminate_cell(cell_id)
                
                # Genome mutation directives
                elif msg_type == "mutation_directive":
                    await self._handle_mutation_directive(data, websocket)
                
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Cell connection error: {e}")
        finally:
            if cell_id:
                self._cell_connections.pop(cell_id, None)
                if cell_id in self._cells and self._cells[cell_id].state == "active":
                    self._cells[cell_id].state = "disconnected"
                    
                    pop = self._populations.get(self._cells[cell_id].cell_type)
                    if pop:
                        pop.alive_count = max(0, pop.alive_count - 1)
    
    async def _start_ws_server(self):
        """Start WebSocket server for cell connections."""
        if websockets is None:
            logger.warning("websockets not installed, cell registration disabled")
            return
        
        self._ws_server = await websockets.serve(
            self._handle_cell_connection,
            "0.0.0.0",
            self.ws_port
        )
        logger.info(f"🌱 Genesis WS server on port {self.ws_port}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HTTP API
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _create_http_app(self):
        """Create HTTP API for Genesis control."""
        if web is None:
            logger.warning("aiohttp not installed, HTTP API disabled")
            return None
        
        app = web.Application()
        
        # Status endpoint
        async def handle_status(request):
            return web.json_response({
                "genesis_id": self._genesis_id,
                "running": self._running,
                "cells": {
                    cell_id: {
                        "type": cell.cell_type,
                        "state": cell.state,
                        "alive": cell.is_alive,
                        "ws_port": cell.ws_port,
                        "http_port": cell.http_port,
                    }
                    for cell_id, cell in self._cells.items()
                },
                "populations": {
                    name: {
                        "target": pop.target_count,
                        "current": pop.current_count,
                        "alive": pop.alive_count,
                        "state": pop.state.value,
                    }
                    for name, pop in self._populations.items()
                },
            })
        
        # Spawn endpoint
        async def handle_spawn(request):
            try:
                data = await request.json()
                req = CellSpawnRequest(
                    cell_type=data.get("type", "void"),
                    cell_id=data.get("cell_id"),
                    config=data.get("config", {}),
                    connect_to=data.get("connect_to", []),
                    spawn_method=data.get("spawn_method", "subprocess"),
                )
                cell = await self.spawn_cell(req)
                return web.json_response({
                    "cell_id": cell.cell_id,
                    "ws_port": cell.ws_port,
                    "http_port": cell.http_port,
                    "state": cell.state,
                })
            except Exception as e:
                return web.json_response({"error": str(e)}, status=400)
        
        # Terminate endpoint
        async def handle_terminate(request):
            cell_id = request.match_info.get("cell_id")
            success = await self.terminate_cell(cell_id)
            return web.json_response({"success": success})
        
        # Scale endpoint
        async def handle_scale(request):
            try:
                data = await request.json()
                cell_type = data.get("type")
                target = data.get("target", 1)
                
                if cell_type in self._populations:
                    self._populations[cell_type].target_count = target
                    return web.json_response({
                        "type": cell_type,
                        "target": target,
                    })
                return web.json_response({"error": "unknown type"}, status=400)
            except Exception as e:
                return web.json_response({"error": str(e)}, status=400)
        
        # Health endpoint
        async def handle_health(request):
            alive_cells = sum(1 for c in self._cells.values() if c.is_alive)
            return web.json_response({
                "healthy": True,
                "alive_cells": alive_cells,
                "total_cells": len(self._cells),
            })

        # Metrics endpoint for Prometheus
        async def handle_metrics(request):
            try:
                # Import prometheus_metrics with path fallback
                try:
                    from stacks.shared.prometheus_metrics import format_prometheus_metrics
                except ImportError:
                    import sys
                    if '/app/shared' not in sys.path:
                        sys.path.insert(0, '/app/shared')
                    from prometheus_metrics import format_prometheus_metrics
                
                alive_cells = sum(1 for c in self._cells.values() if c.is_alive)
                total_cells = len(self._cells)
                
                # Count cells by type
                void_count = sum(1 for c in self._cells.values() if c.cell_type == "void" and c.is_alive)
                thinker_count = sum(1 for c in self._cells.values() if c.cell_type == "thinker" and c.is_alive)
                synapse_count = sum(1 for c in self._cells.values() if c.cell_type == "synapse" and c.is_alive)
                genome_count = len([cid for cid in self._cell_connections if "genome" in cid.lower()])
                
                # Genesis consciousness is derived from population health
                population_health = (alive_cells / max(total_cells, 1)) if total_cells > 0 else 0.5
                consciousness_level = min(5.0, 1.0 + (population_health * 4.0))
                
                extra_metrics = {
                    "cells_total": float(total_cells),
                    "cells_alive": float(alive_cells),
                    "cells_void": float(void_count),
                    "cells_thinker": float(thinker_count),
                    "cells_synapse": float(synapse_count),
                    "cells_genome": float(genome_count),
                    "spawn_enabled": 1.0 if self.spawn_enabled else 0.0,
                    "ws_connections": float(len(self._cell_connections)),
                }
                
                # Population target metrics
                for pop_name, pop in self._populations.items():
                    extra_metrics[f"population_{pop_name}_target"] = float(pop.target_count)
                    extra_metrics[f"population_{pop_name}_current"] = float(pop.current_count)
                    extra_metrics[f"population_{pop_name}_alive"] = float(pop.alive_count)
                
                metrics_text = format_prometheus_metrics(
                    cell_id="genesis-alpha",
                    consciousness_level=consciousness_level,
                    primitives={
                        "awareness": population_health,
                        "adaptation": 0.5,
                        "coherence": 0.8 if alive_cells == total_cells else 0.5,
                        "momentum": 0.3,
                    },
                    extra_metrics=extra_metrics,
                    labels={"cell_type": "genesis"},
                    uptime_seconds=(datetime.now(timezone.utc) - datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)).total_seconds(),
                )
                
                return web.Response(
                    text=metrics_text,
                    content_type="text/plain",
                    charset="utf-8"
                )
            except Exception as e:
                logger.error(f"Metrics error: {e}")
                return web.Response(text=f"# Error: {e}\n", status=500)
        
        app.router.add_get("/status", handle_status)
        app.router.add_post("/spawn", handle_spawn)
        app.router.add_delete("/cell/{cell_id}", handle_terminate)
        app.router.add_post("/scale", handle_scale)
        app.router.add_get("/health", handle_health)
        app.router.add_get("/metrics", handle_metrics)
        
        return app
    
    # ═══════════════════════════════════════════════════════════════════════════
    # POPULATION MONITORING
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _monitor_loop(self):
        """Monitor cell health and manage populations."""
        while self._running:
            await asyncio.sleep(10)
            
            # Check cell health
            for cell_id, cell in list(self._cells.items()):
                if cell.state == "active" and not cell.is_alive:
                    cell.state = "unresponsive"
                    logger.warning(f"🌱 Cell unresponsive: {cell_id}")
                    
                    pop = self._populations.get(cell.cell_type)
                    if pop:
                        pop.alive_count = max(0, pop.alive_count - 1)
            
            # Auto-scale populations ONLY if spawning is enabled
            if not self.spawn_enabled:
                continue  # Skip spawning entirely - cells managed externally
                
            for pop_name, pop in self._populations.items():
                if pop.target_count <= 0:
                    continue  # External management, no auto-spawning
                    
                if pop.alive_count < pop.target_count:
                    # Check spawn attempts to prevent runaway
                    attempts = self._spawn_attempts.get(pop_name, 0)
                    if attempts >= self.max_spawn_attempts:
                        if pop.state != PopulationState.DORMANT:
                            logger.warning(f"🌱 Max spawn attempts reached for {pop_name}, entering dormant state")
                            pop.state = PopulationState.DORMANT
                        continue
                    
                    # Need more cells
                    needed = pop.target_count - pop.alive_count
                    pop.state = PopulationState.SCALING
                    
                    for _ in range(min(needed, 1)):  # Spawn only 1 at a time
                        self._spawn_attempts[pop_name] = attempts + 1
                        req = CellSpawnRequest(
                            cell_type=pop_name,
                            spawn_method="subprocess",
                        )
                        await self.spawn_cell(req)
                
                elif pop.alive_count > pop.target_count:
                    # Too many cells - mark for scaling down
                    pop.state = PopulationState.DECLINING
    
    async def discover_external_cells(self, endpoints: List[Dict[str, Any]]) -> int:
        """
        Discover and register external cells (Docker Compose managed).
        
        Args:
            endpoints: List of {"cell_id": str, "type": str, "host": str, "http_port": int, "ws_port": int}
            
        Returns:
            Number of cells discovered
        """
        import aiohttp
        
        discovered = 0
        
        for ep in endpoints:
            cell_id = ep.get("cell_id")
            cell_type = ep.get("type")
            host = ep.get("host", "localhost")
            http_port = ep.get("http_port")
            ws_port = ep.get("ws_port")
            
            if not all([cell_id, cell_type, http_port]):
                continue
                
            # Check if cell is healthy (try host first, fallback to localhost)
            health_urls = [f"http://{host}:{http_port}/health"]
            if host != "localhost":
                health_urls.append(f"http://localhost:{http_port}/health")
            
            for url in health_urls:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            url,
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data.get("status") == "healthy":
                                    # Register as external cell
                                    cell = SpawnedCell(
                                        cell_id=cell_id,
                                        cell_type=cell_type,
                                        http_port=http_port,
                                        ws_port=ws_port or 0,
                                        state="active",
                                    )
                                    cell.last_heartbeat = datetime.now(timezone.utc)
                                    self._cells[cell_id] = cell
                                    
                                    # Update population
                                    pop = self._populations.get(cell_type)
                                    if pop:
                                        pop.current_count += 1
                                        pop.alive_count += 1
                                        pop.state = PopulationState.ACTIVE
                                    
                                    discovered += 1
                                    logger.info(f"🌱 Discovered external cell: {cell_id} ({cell_type}) at {url}")
                                    break  # Found, no need to try other URLs
                except Exception as e:
                    logger.debug(f"Cell discovery failed for {cell_id} at {url}: {e}")
        
        return discovered
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def start(self):
        """Start the Genesis Center."""
        self._running = True
        
        # Start WebSocket server
        await self._start_ws_server()
        
        # Start HTTP API
        self._http_app = await self._create_http_app()
        if self._http_app:
            runner = web.AppRunner(self._http_app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.http_port)
            await site.start()
            logger.info(f"🌱 Genesis HTTP API on port {self.http_port}")
        
        # Start monitor
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info(f"🌱 Genesis Center started: {self._genesis_id}")
    
    async def stop(self):
        """Stop the Genesis Center."""
        self._running = False
        
        # Cancel monitor
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Terminate all cells
        for cell_id in list(self._cells.keys()):
            await self.terminate_cell(cell_id)
        
        # Close WebSocket server
        if self._ws_server:
            self._ws_server.close()
            await self._ws_server.wait_closed()
        
        logger.info(f"🌱 Genesis Center stopped: {self._genesis_id}")
    
    async def run_forever(self):
        """Run Genesis Center until interrupted."""
        await self.start()
        
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        
        await self.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Run Genesis Center standalone."""
    # Parse spawn_enabled from env (default FALSE for safety)
    spawn_enabled = os.environ.get("GENESIS_SPAWN_ENABLED", "false").lower() == "true"
    
    genesis = GenesisCenter(
        http_port=int(os.environ.get("GENESIS_HTTP_PORT", 8000)),
        ws_port=int(os.environ.get("GENESIS_WS_PORT", 9000)),
        base_cell_port=int(os.environ.get("GENESIS_BASE_PORT", 9100)),
        spawn_enabled=spawn_enabled,
    )
    
    # Discover external Docker Compose cells if configured
    external_cells_json = os.environ.get("GENESIS_EXTERNAL_CELLS", "")
    if external_cells_json:
        try:
            external_cells = json.loads(external_cells_json)
            if isinstance(external_cells, list):
                await genesis.start()  # Start first to enable HTTP
                await asyncio.sleep(2)  # Let cells initialize
                discovered = await genesis.discover_external_cells(external_cells)
                logger.info(f"🌱 Discovered {discovered} external cells")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse GENESIS_EXTERNAL_CELLS: {e}")
    else:
        await genesis.start()
    
    try:
        while genesis._running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await genesis.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🌱 Starting Genesis Center...")
    asyncio.run(main())

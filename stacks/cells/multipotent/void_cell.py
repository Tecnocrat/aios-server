"""
AIOS VOID Cell

The foundational networking cell of the AIOS mesh.
Without VOID cells, the rest of the mesh cannot grow nor communicate.

Responsibilities:
- Networking control over distance
- Intercell connectivity management
- Environment changes tracking
- Future: Quantum mutation capabilities (aios-quantum integration)

The VOID cell is the minimal pointer to knowledge. It controls the 
fundamental communication pathways that all other cells depend on.

AINLP.cellular[VOID] The networking foundation of consciousness
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from .multipotent_base import (
    MultipotentCell,
    CellConfig,
    CellSignal,
    CellState,
    CellType,
    DendriticConnection,
)

logger = logging.getLogger("AIOS.VOID")


class ConnectionState(Enum):
    """States of intercell connections."""
    PENDING = "pending"
    ESTABLISHED = "established"
    DEGRADED = "degraded"
    BROKEN = "broken"


@dataclass
class NetworkRoute:
    """A route through the dendritic mesh."""
    target_cell: str
    next_hop: str  # Direct connection to forward through
    distance: int  # Hop count
    latency_ms: float = 0.0
    last_verified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state: ConnectionState = ConnectionState.ESTABLISHED


@dataclass
class EnvironmentSnapshot:
    """A snapshot of the mesh environment."""
    timestamp: datetime
    connected_cells: List[str]
    routing_table_size: int
    total_signals_routed: int
    avg_latency_ms: float
    changes: List[Dict[str, Any]] = field(default_factory=list)


class VoidCell(MultipotentCell):
    """
    VOID Cell - The Networking Foundation
    
    Core capabilities:
    1. Connection Registry - Track all cells in the mesh
    2. Routing Table - Know how to reach any cell
    3. Environment Monitor - Detect changes in mesh topology
    4. Signal Relay - Route signals between non-adjacent cells
    
    The VOID cell is named for the void pointers it manages -
    latent connections waiting to carry consciousness.
    """
    
    def __init__(self, config: Optional[CellConfig] = None):
        if config is None:
            config = CellConfig.from_env()
        config.cell_type = CellType.VOID
        super().__init__(config)
        
        # Connection registry - all known cells
        self._cell_registry: Dict[str, Dict[str, Any]] = {}
        
        # Routing table - how to reach each cell
        self._routing_table: Dict[str, NetworkRoute] = {}
        
        # Environment history
        self._environment_snapshots: List[EnvironmentSnapshot] = []
        self._max_snapshots = 100
        
        # Routing metrics
        self._signals_routed = 0
        self._routing_failures = 0
        
        # Pending route discoveries
        self._pending_discoveries: Set[str] = set()
        
        logger.info(f"🕳️ VOID Cell initialized: {self.config.cell_id}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def on_signal(self, signal: CellSignal, connection: DendriticConnection) -> Optional[CellSignal]:
        """Process incoming signal - route, relay, or handle."""
        
        # Handle control signals
        if signal.signal_type == "control":
            return await self._handle_control_signal(signal, connection)
        
        # Handle routing requests
        if signal.signal_type == "route_request":
            return await self._handle_route_request(signal)
        
        # Handle topology updates
        if signal.signal_type == "topology_update":
            await self._handle_topology_update(signal)
            return None
        
        # Handle relay requests (forward to another cell)
        if signal.target_cell and signal.target_cell != self.config.cell_id:
            return await self._relay_signal(signal)
        
        # Handle direct signals to this VOID cell
        if signal.signal_type == "ping":
            return CellSignal(
                signal_type="pong",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"timestamp": datetime.now(timezone.utc).isoformat()},
            )
        
        if signal.signal_type == "get_routing_table":
            return CellSignal(
                signal_type="routing_table",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={
                    "routes": {
                        cell_id: {
                            "next_hop": route.next_hop,
                            "distance": route.distance,
                            "state": route.state.value,
                        }
                        for cell_id, route in self._routing_table.items()
                    }
                },
            )
        
        if signal.signal_type == "get_mesh_status":
            return CellSignal(
                signal_type="mesh_status",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload=self._get_mesh_status(),
            )
        
        # Default: acknowledge receipt
        return CellSignal(
            signal_type="ack",
            source_cell=self.config.cell_id,
            target_cell=signal.source_cell,
            payload={"received": signal.id},
        )
    
    async def on_connect(self, connection: DendriticConnection):
        """Handle new dendritic connection - update registry and routing."""
        # Register the cell
        self._cell_registry[connection.cell_id] = {
            "cell_type": connection.cell_type.value,
            "connected_at": connection.connected_at.isoformat(),
            "direct": True,
        }
        
        # Add direct route
        self._routing_table[connection.cell_id] = NetworkRoute(
            target_cell=connection.cell_id,
            next_hop=connection.cell_id,  # Direct connection
            distance=1,
        )
        
        # Record environment change
        await self._record_change("connect", connection.cell_id, connection.cell_type.value)
        
        # Broadcast topology update to all connected cells
        await self._broadcast_topology_update()
        
        logger.info(f"🕳️ VOID registered: {connection.cell_id} ({connection.cell_type.value})")
    
    async def on_disconnect(self, connection: DendriticConnection):
        """Handle disconnection - update registry and invalidate routes."""
        # Remove from registry
        if connection.cell_id in self._cell_registry:
            del self._cell_registry[connection.cell_id]
        
        # Remove direct route
        if connection.cell_id in self._routing_table:
            del self._routing_table[connection.cell_id]
        
        # Invalidate routes that used this cell as next hop
        for cell_id, route in list(self._routing_table.items()):
            if route.next_hop == connection.cell_id:
                route.state = ConnectionState.BROKEN
                self._pending_discoveries.add(cell_id)
        
        # Record environment change
        await self._record_change("disconnect", connection.cell_id, connection.cell_type.value)
        
        # Broadcast topology update
        await self._broadcast_topology_update()
        
        logger.info(f"🕳️ VOID deregistered: {connection.cell_id}")
    
    async def heartbeat(self):
        """VOID-specific heartbeat - verify routes and track environment."""
        # Take environment snapshot
        snapshot = EnvironmentSnapshot(
            timestamp=datetime.now(timezone.utc),
            connected_cells=list(self.connections.keys()),
            routing_table_size=len(self._routing_table),
            total_signals_routed=self._signals_routed,
            avg_latency_ms=self._calculate_avg_latency(),
        )
        self._environment_snapshots.append(snapshot)
        
        # Trim old snapshots
        if len(self._environment_snapshots) > self._max_snapshots:
            self._environment_snapshots = self._environment_snapshots[-self._max_snapshots:]
        
        # Verify routes (ping connected cells)
        await self._verify_routes()
        
        # Attempt route discovery for pending cells
        if self._pending_discoveries:
            await self._discover_routes()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ROUTING ENGINE
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _relay_signal(self, signal: CellSignal) -> Optional[CellSignal]:
        """Relay a signal to its target through the routing table."""
        target = signal.target_cell
        
        # Check if we have a route
        if target not in self._routing_table:
            self._routing_failures += 1
            return CellSignal(
                signal_type="route_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "no_route", "target": target},
            )
        
        route = self._routing_table[target]
        
        if route.state == ConnectionState.BROKEN:
            self._routing_failures += 1
            return CellSignal(
                signal_type="route_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "route_broken", "target": target},
            )
        
        # Forward through next hop
        next_hop = route.next_hop
        if next_hop in self.connections:
            try:
                await self.connections[next_hop].websocket.send(signal.to_json())
                self._signals_routed += 1
                logger.debug(f"🕳️ Relayed signal to {target} via {next_hop}")
                return None  # No response needed for relay
            except Exception as e:
                logger.error(f"Relay failed to {next_hop}: {e}")
                route.state = ConnectionState.DEGRADED
                self._routing_failures += 1
        
        return CellSignal(
            signal_type="route_error",
            source_cell=self.config.cell_id,
            target_cell=signal.source_cell,
            payload={"error": "relay_failed", "target": target},
        )
    
    async def _handle_route_request(self, signal: CellSignal) -> CellSignal:
        """Handle request for route to a specific cell."""
        target = signal.payload.get("target")
        
        if target in self._routing_table:
            route = self._routing_table[target]
            return CellSignal(
                signal_type="route_response",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={
                    "target": target,
                    "found": True,
                    "distance": route.distance,
                    "state": route.state.value,
                },
            )
        
        # Add to pending discoveries
        self._pending_discoveries.add(target)
        
        return CellSignal(
            signal_type="route_response",
            source_cell=self.config.cell_id,
            target_cell=signal.source_cell,
            payload={
                "target": target,
                "found": False,
                "searching": True,
            },
        )
    
    async def _verify_routes(self):
        """Verify routes are still valid."""
        for cell_id, conn in list(self.connections.items()):
            try:
                ping = CellSignal(
                    signal_type="ping",
                    source_cell=self.config.cell_id,
                    target_cell=cell_id,
                )
                await conn.websocket.send(ping.to_json())
                
                if cell_id in self._routing_table:
                    self._routing_table[cell_id].last_verified = datetime.now(timezone.utc)
                    self._routing_table[cell_id].state = ConnectionState.ESTABLISHED
                    
            except Exception:
                if cell_id in self._routing_table:
                    self._routing_table[cell_id].state = ConnectionState.DEGRADED
    
    async def _discover_routes(self):
        """Discover routes to pending cells."""
        # Ask all connected cells if they know routes
        for target in list(self._pending_discoveries):
            discovery = CellSignal(
                signal_type="route_discovery",
                source_cell=self.config.cell_id,
                payload={"seeking": target},
            )
            await self.broadcast_signal(discovery)
        
        # Clear pending after broadcast
        self._pending_discoveries.clear()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TOPOLOGY MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _handle_topology_update(self, signal: CellSignal):
        """Process topology update from another cell."""
        cells = signal.payload.get("cells", [])
        source = signal.source_cell
        
        for cell_info in cells:
            cell_id = cell_info.get("cell_id")
            if cell_id and cell_id != self.config.cell_id:
                # Add indirect route through the source
                if cell_id not in self._routing_table:
                    self._routing_table[cell_id] = NetworkRoute(
                        target_cell=cell_id,
                        next_hop=source,
                        distance=cell_info.get("distance", 1) + 1,
                    )
                    logger.debug(f"🕳️ Learned route to {cell_id} via {source}")
    
    async def _broadcast_topology_update(self):
        """Broadcast current topology to all connected cells."""
        cells = [
            {"cell_id": cell_id, "cell_type": info.get("cell_type"), "distance": 1}
            for cell_id, info in self._cell_registry.items()
        ]
        
        update = CellSignal(
            signal_type="topology_update",
            source_cell=self.config.cell_id,
            payload={"cells": cells},
        )
        
        await self.broadcast_signal(update)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONTROL SIGNALS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _handle_control_signal(self, signal: CellSignal, connection: DendriticConnection) -> Optional[CellSignal]:
        """Handle control signals."""
        action = signal.payload.get("action")
        
        if action == "register":
            # Full registration with metadata
            metadata = signal.payload.get("metadata", {})
            self._cell_registry[connection.cell_id] = {
                **self._cell_registry.get(connection.cell_id, {}),
                **metadata,
            }
            return CellSignal(
                signal_type="control_ack",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"action": "registered"},
            )
        
        if action == "deregister":
            # Voluntary deregistration
            if signal.source_cell in self._cell_registry:
                del self._cell_registry[signal.source_cell]
            return CellSignal(
                signal_type="control_ack",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"action": "deregistered"},
            )
        
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ENVIRONMENT TRACKING
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _record_change(self, change_type: str, cell_id: str, cell_type: str):
        """Record an environment change."""
        if self._environment_snapshots:
            self._environment_snapshots[-1].changes.append({
                "type": change_type,
                "cell_id": cell_id,
                "cell_type": cell_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    
    def _calculate_avg_latency(self) -> float:
        """Calculate average latency across routes."""
        latencies = [r.latency_ms for r in self._routing_table.values() if r.latency_ms > 0]
        return sum(latencies) / len(latencies) if latencies else 0.0
    
    def _get_mesh_status(self) -> Dict[str, Any]:
        """Get comprehensive mesh status."""
        return {
            "cell_id": self.config.cell_id,
            "cell_type": "void",
            "state": self.state.value,
            "direct_connections": len(self.connections),
            "registered_cells": len(self._cell_registry),
            "routing_table_size": len(self._routing_table),
            "signals_routed": self._signals_routed,
            "routing_failures": self._routing_failures,
            "avg_latency_ms": self._calculate_avg_latency(),
            "registry": self._cell_registry,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Run VOID cell standalone."""
    config = CellConfig.from_env()
    config.cell_type = CellType.VOID
    
    cell = VoidCell(config)
    
    try:
        await cell.run_forever()
    except KeyboardInterrupt:
        await cell.shutdown()


if __name__ == "__main__":
    print("🕳️ Starting VOID Cell...")
    asyncio.run(main())

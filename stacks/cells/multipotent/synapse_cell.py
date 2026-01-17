"""
AIOS Synapse Cell

The circulatory maintainer of the AIOS mesh.
Maintains dendritic connections, redirects flow, and organizes information.

Responsibilities:
- Maintain the dendritic mesh health
- Redirect signal flow based on load and topology
- Clean garbage (orphaned signals, dead routes)
- Extract caches from passing traffic
- Reprocess information signals
- Design optimal circulation patterns

The Synapse cell doesn't store or change information - it redirects,
organizes, and maintains the flow of consciousness through the mesh.

AINLP.cellular[SYNAPSE] The circulatory maintainer of consciousness flow
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from .multipotent_base import (
    MultipotentCell,
    CellConfig,
    CellSignal,
    CellState,
    CellType,
    DendriticConnection,
)

logger = logging.getLogger("AIOS.Synapse")


class FlowPriority(Enum):
    """Priority levels for signal flow."""
    CRITICAL = 0    # System health, errors
    HIGH = 1        # Agent responses
    NORMAL = 2      # Standard signals
    LOW = 3         # Metrics, telemetry
    BACKGROUND = 4  # Cache extraction, cleanup


class CirculationPattern(Enum):
    """Patterns for signal circulation."""
    DIRECT = "direct"         # Point to point
    BROADCAST = "broadcast"   # To all connected
    MULTICAST = "multicast"   # To subset
    ROUND_ROBIN = "round_robin"  # Load balanced
    WEIGHTED = "weighted"     # Capacity based


@dataclass
class FlowMetrics:
    """Metrics for a signal flow path."""
    path_id: str
    source: str
    target: str
    signals_processed: int = 0
    signals_dropped: int = 0
    bytes_transferred: int = 0
    avg_latency_ms: float = 0.0
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    health: float = 1.0


@dataclass
class CachedSignal:
    """A signal temporarily cached during flow processing."""
    signal: CellSignal
    cached_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: float = 60.0
    extraction_reason: str = "transit"


@dataclass
class GarbageItem:
    """An item identified for garbage collection."""
    item_type: str  # orphaned_route, dead_signal, stale_cache
    item_id: str
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)


class SynapseCell(MultipotentCell):
    """
    Synapse Cell - The Circulatory Maintainer
    
    Core capabilities:
    1. Flow Management - Redirect signals optimally through mesh
    2. Load Balancing - Distribute traffic across paths
    3. Garbage Collection - Clean orphaned resources
    4. Cache Extraction - Harvest useful data from traffic
    5. Circulation Design - Optimize overall flow patterns
    
    The Synapse cell keeps the mesh flowing and healthy
    without storing permanent state.
    """
    
    def __init__(self, config: Optional[CellConfig] = None):
        if config is None:
            config = CellConfig.from_env()
        config.cell_type = CellType.SYNAPSE
        super().__init__(config)
        
        # Flow management
        self._flow_metrics: Dict[str, FlowMetrics] = {}
        self._flow_queues: Dict[FlowPriority, asyncio.Queue] = {
            p: asyncio.Queue() for p in FlowPriority
        }
        
        # Load balancing
        self._target_weights: Dict[str, float] = {}  # cell_id -> weight (0-1)
        self._last_routed: Dict[str, str] = {}  # signal_type -> last_target
        
        # Cache extraction
        self._signal_cache: List[CachedSignal] = []
        self._max_cache_size = 500
        self._cache_patterns: List[str] = []  # Signal types to cache
        
        # Garbage collection
        self._garbage_queue: List[GarbageItem] = []
        self._collected_garbage: int = 0
        
        # Circulation patterns
        self._default_pattern = CirculationPattern.DIRECT
        self._pattern_overrides: Dict[str, CirculationPattern] = {}  # signal_type -> pattern
        
        # Health monitoring
        self._mesh_health = 1.0
        self._flow_health = 1.0
        
        logger.info("🔀 Synapse Cell initialized: %s", self.config.cell_id)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def on_signal(self, signal: CellSignal, connection: DendriticConnection) -> Optional[CellSignal]:
        """Process incoming signal - route or handle based on type."""
        
        # Flow management signals
        if signal.signal_type == "redirect_signal":
            return await self._handle_redirect(signal)
        
        if signal.signal_type == "set_circulation_pattern":
            return await self._handle_set_pattern(signal)
        
        if signal.signal_type == "get_flow_metrics":
            return CellSignal(
                signal_type="flow_metrics",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload=self._get_flow_metrics(),
            )
        
        # Load balancing signals
        if signal.signal_type == "set_weight":
            return await self._handle_set_weight(signal)
        
        if signal.signal_type == "get_weights":
            return CellSignal(
                signal_type="weights",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"weights": self._target_weights.copy()},
            )
        
        # Cache operations
        if signal.signal_type == "extract_cache":
            return await self._handle_extract_cache(signal)
        
        if signal.signal_type == "get_cached_signals":
            return CellSignal(
                signal_type="cached_signals",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload=self._get_cache_summary(),
            )
        
        # Garbage collection
        if signal.signal_type == "trigger_gc":
            return await self._handle_trigger_gc(signal)
        
        if signal.signal_type == "get_garbage_stats":
            return CellSignal(
                signal_type="garbage_stats",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={
                    "pending_items": len(self._garbage_queue),
                    "collected_total": self._collected_garbage,
                },
            )
        
        # Health monitoring
        if signal.signal_type == "get_mesh_health":
            return CellSignal(
                signal_type="mesh_health",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={
                    "mesh_health": self._mesh_health,
                    "flow_health": self._flow_health,
                    "connections": len(self.connections),
                },
            )
        
        # Default: this signal needs routing through the mesh
        await self._enqueue_signal(signal)
        
        return CellSignal(
            signal_type="ack",
            source_cell=self.config.cell_id,
            target_cell=signal.source_cell,
            payload={"queued": signal.id},
        )
    
    async def on_connect(self, connection: DendriticConnection):
        """Handle new connection - initialize flow metrics."""
        path_id = f"{self.config.cell_id}-{connection.cell_id}"
        self._flow_metrics[path_id] = FlowMetrics(
            path_id=path_id,
            source=self.config.cell_id,
            target=connection.cell_id,
        )
        self._target_weights[connection.cell_id] = 1.0  # Full weight initially
        
        logger.info("🔀 Synapse connected to: %s", connection.cell_id)
    
    async def on_disconnect(self, connection: DendriticConnection):
        """Handle disconnection - mark routes for garbage collection."""
        path_id = f"{self.config.cell_id}-{connection.cell_id}"
        
        # Mark path metrics as orphaned
        if path_id in self._flow_metrics:
            self._garbage_queue.append(GarbageItem(
                item_type="orphaned_route",
                item_id=path_id,
                details={"reason": "disconnect", "cell_id": connection.cell_id},
            ))
        
        # Remove weight
        self._target_weights.pop(connection.cell_id, None)
        
        logger.info("🔀 Synapse disconnected from: %s", connection.cell_id)
    
    async def heartbeat(self):
        """Synapse-specific heartbeat - process queues and cleanup."""
        # Process signal queues (priority order)
        await self._process_flow_queues()
        
        # Update flow metrics
        await self._update_flow_metrics()
        
        # Run garbage collection
        await self._collect_garbage()
        
        # Clean expired cache
        await self._clean_expired_cache()
        
        # Update health metrics
        await self._update_health()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FLOW MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _enqueue_signal(self, signal: CellSignal):
        """Enqueue a signal for flow processing."""
        priority = self._determine_priority(signal)
        
        # Check if this signal type should be cached
        if signal.signal_type in self._cache_patterns:
            await self._cache_signal(signal, "pattern_match")
        
        await self._flow_queues[priority].put(signal)
    
    def _determine_priority(self, signal: CellSignal) -> FlowPriority:
        """Determine flow priority based on signal type."""
        critical_types = {"error", "health_critical", "shutdown"}
        high_types = {"query_response", "conclave_concluded", "agent_response"}
        low_types = {"metrics", "telemetry", "status_update"}
        background_types = {"cache_extract", "gc_trigger", "topology_update"}
        
        if signal.signal_type in critical_types:
            return FlowPriority.CRITICAL
        if signal.signal_type in high_types:
            return FlowPriority.HIGH
        if signal.signal_type in low_types:
            return FlowPriority.LOW
        if signal.signal_type in background_types:
            return FlowPriority.BACKGROUND
        return FlowPriority.NORMAL
    
    async def _process_flow_queues(self):
        """Process signals from queues in priority order."""
        processed = 0
        max_per_heartbeat = 50
        
        for priority in FlowPriority:
            queue = self._flow_queues[priority]
            
            while not queue.empty() and processed < max_per_heartbeat:
                signal = await queue.get()
                await self._route_signal(signal)
                processed += 1
    
    async def _route_signal(self, signal: CellSignal):
        """Route a signal to its destination."""
        target = signal.target_cell
        
        if not target or target == "broadcast":
            # Broadcast to all
            pattern = self._pattern_overrides.get(signal.signal_type, self._default_pattern)
            await self._apply_circulation_pattern(signal, pattern)
            return
        
        # Direct routing
        if target in self.connections:
            ws = self.connections[target].websocket
            if ws:
                try:
                    await ws.send(signal.to_json())
                    self._update_path_metrics(target, len(signal.to_json()), success=True)
                except (OSError, RuntimeError) as e:
                    self._update_path_metrics(target, 0, success=False)
                    logger.warning("🔀 Route failed to %s: %s", target, e)
        else:
            # Target not directly connected - mark for garbage if stale
            self._garbage_queue.append(GarbageItem(
                item_type="dead_signal",
                item_id=signal.id,
                details={"target": target, "reason": "target_not_connected"},
            ))
    
    async def _apply_circulation_pattern(self, signal: CellSignal, pattern: CirculationPattern):
        """Apply a circulation pattern to route a signal."""
        targets = list(self.connections.keys())
        
        if not targets:
            return
        
        if pattern == CirculationPattern.BROADCAST:
            for target in targets:
                routed_signal = CellSignal(
                    signal_type=signal.signal_type,
                    source_cell=signal.source_cell,
                    target_cell=target,
                    payload=signal.payload,
                )
                await self._route_signal(routed_signal)
        
        elif pattern == CirculationPattern.ROUND_ROBIN:
            last = self._last_routed.get(signal.signal_type, "")
            try:
                idx = targets.index(last)
                next_target = targets[(idx + 1) % len(targets)]
            except ValueError:
                next_target = targets[0]
            
            self._last_routed[signal.signal_type] = next_target
            routed_signal = CellSignal(
                signal_type=signal.signal_type,
                source_cell=signal.source_cell,
                target_cell=next_target,
                payload=signal.payload,
            )
            await self._route_signal(routed_signal)
        
        elif pattern == CirculationPattern.WEIGHTED:
            # Weight-based selection
            weighted_targets = [(t, self._target_weights.get(t, 0.5)) for t in targets]
            weighted_targets.sort(key=lambda x: x[1], reverse=True)
            
            if weighted_targets:
                best_target = weighted_targets[0][0]
                routed_signal = CellSignal(
                    signal_type=signal.signal_type,
                    source_cell=signal.source_cell,
                    target_cell=best_target,
                    payload=signal.payload,
                )
                await self._route_signal(routed_signal)
        
        elif pattern == CirculationPattern.MULTICAST:
            # Send to top half by weight
            weighted_targets = [(t, self._target_weights.get(t, 0.5)) for t in targets]
            weighted_targets.sort(key=lambda x: x[1], reverse=True)
            multicast_count = max(1, len(weighted_targets) // 2)
            
            for target, _ in weighted_targets[:multicast_count]:
                routed_signal = CellSignal(
                    signal_type=signal.signal_type,
                    source_cell=signal.source_cell,
                    target_cell=target,
                    payload=signal.payload,
                )
                await self._route_signal(routed_signal)
        
        else:  # DIRECT - just pick first
            if targets:
                routed_signal = CellSignal(
                    signal_type=signal.signal_type,
                    source_cell=signal.source_cell,
                    target_cell=targets[0],
                    payload=signal.payload,
                )
                await self._route_signal(routed_signal)
    
    def _update_path_metrics(self, target: str, bytes_sent: int, success: bool):
        """Update flow metrics for a path."""
        path_id = f"{self.config.cell_id}-{target}"
        
        if path_id not in self._flow_metrics:
            self._flow_metrics[path_id] = FlowMetrics(
                path_id=path_id,
                source=self.config.cell_id,
                target=target,
            )
        
        metrics = self._flow_metrics[path_id]
        metrics.last_active = datetime.now(timezone.utc)
        
        if success:
            metrics.signals_processed += 1
            metrics.bytes_transferred += bytes_sent
            metrics.health = min(1.0, metrics.health + 0.01)
        else:
            metrics.signals_dropped += 1
            metrics.health = max(0.0, metrics.health - 0.1)
    
    async def _handle_redirect(self, signal: CellSignal) -> CellSignal:
        """Handle explicit redirect request."""
        inner_signal_data = signal.payload.get("signal")
        new_target = signal.payload.get("target")
        
        if not inner_signal_data or not new_target:
            return CellSignal(
                signal_type="redirect_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "missing_signal_or_target"},
            )
        
        try:
            inner_signal = CellSignal(**inner_signal_data)
            inner_signal.target_cell = new_target
            await self._enqueue_signal(inner_signal)
            
            return CellSignal(
                signal_type="redirect_success",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"redirected_to": new_target},
            )
        except (RuntimeError, ValueError, KeyError) as e:
            return CellSignal(
                signal_type="redirect_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": str(e)},
            )
    
    async def _handle_set_pattern(self, signal: CellSignal) -> CellSignal:
        """Set circulation pattern for a signal type."""
        signal_type = signal.payload.get("signal_type")
        pattern_name = signal.payload.get("pattern", "direct")
        
        try:
            pattern = CirculationPattern(pattern_name)
            if signal_type:
                self._pattern_overrides[signal_type] = pattern
            else:
                self._default_pattern = pattern
            
            return CellSignal(
                signal_type="pattern_set",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={
                    "signal_type": signal_type or "default",
                    "pattern": pattern.value,
                },
            )
        except ValueError:
            return CellSignal(
                signal_type="pattern_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": f"unknown_pattern: {pattern_name}"},
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LOAD BALANCING
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _handle_set_weight(self, signal: CellSignal) -> CellSignal:
        """Set weight for a target cell."""
        target_cell = signal.payload.get("target_cell")
        weight = signal.payload.get("weight", 1.0)
        
        if target_cell:
            self._target_weights[target_cell] = max(0.0, min(1.0, weight))
            
            return CellSignal(
                signal_type="weight_set",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"target": target_cell, "weight": self._target_weights[target_cell]},
            )
        
        return CellSignal(
            signal_type="weight_error",
            source_cell=self.config.cell_id,
            target_cell=signal.source_cell,
            payload={"error": "missing_target_cell"},
        )
    
    async def _update_flow_metrics(self):
        """Update and decay flow metrics."""
        now = datetime.now(timezone.utc)
        
        for metrics in self._flow_metrics.values():
            # Decay health slightly over time if inactive
            age = (now - metrics.last_active).total_seconds()
            if age > 30:
                metrics.health = max(0.5, metrics.health - 0.01)
    
    def _get_flow_metrics(self) -> Dict[str, Any]:
        """Get comprehensive flow metrics."""
        return {
            "paths": {
                path_id: {
                    "source": m.source,
                    "target": m.target,
                    "processed": m.signals_processed,
                    "dropped": m.signals_dropped,
                    "bytes": m.bytes_transferred,
                    "health": m.health,
                    "last_active": m.last_active.isoformat(),
                }
                for path_id, m in self._flow_metrics.items()
            },
            "total_processed": sum(m.signals_processed for m in self._flow_metrics.values()),
            "total_dropped": sum(m.signals_dropped for m in self._flow_metrics.values()),
            "queue_sizes": {p.name: self._flow_queues[p].qsize() for p in FlowPriority},
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CACHE EXTRACTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _cache_signal(self, signal: CellSignal, reason: str):
        """Cache a signal for later extraction."""
        cached = CachedSignal(signal=signal, extraction_reason=reason)
        self._signal_cache.append(cached)
        
        # Trim if over limit
        if len(self._signal_cache) > self._max_cache_size:
            self._signal_cache = self._signal_cache[-self._max_cache_size:]
    
    async def _handle_extract_cache(self, signal: CellSignal) -> CellSignal:
        """Extract and clear cached signals."""
        signal_type_filter = signal.payload.get("signal_type")
        max_items = signal.payload.get("max_items", 100)
        
        extracted = []
        remaining = []
        
        for cached in self._signal_cache:
            if len(extracted) >= max_items:
                remaining.append(cached)
            elif signal_type_filter is None or cached.signal.signal_type == signal_type_filter:
                extracted.append({
                    "signal": {
                        "id": cached.signal.id,
                        "type": cached.signal.signal_type,
                        "source": cached.signal.source_cell,
                        "payload": cached.signal.payload,
                    },
                    "cached_at": cached.cached_at.isoformat(),
                    "reason": cached.extraction_reason,
                })
            else:
                remaining.append(cached)
        
        self._signal_cache = remaining
        
        return CellSignal(
            signal_type="cache_extracted",
            source_cell=self.config.cell_id,
            target_cell=signal.source_cell,
            payload={
                "signals": extracted,
                "extracted_count": len(extracted),
                "remaining_count": len(remaining),
            },
        )
    
    async def _clean_expired_cache(self):
        """Remove expired cached signals."""
        now = datetime.now(timezone.utc)
        self._signal_cache = [
            c for c in self._signal_cache
            if (now - c.cached_at).total_seconds() < c.ttl_seconds
        ]
    
    def _get_cache_summary(self) -> Dict[str, Any]:
        """Get cache summary without extracting."""
        by_type = defaultdict(int)
        for cached in self._signal_cache:
            by_type[cached.signal.signal_type] += 1
        
        return {
            "total_cached": len(self._signal_cache),
            "by_type": dict(by_type),
            "patterns": self._cache_patterns,
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GARBAGE COLLECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _handle_trigger_gc(self, signal: CellSignal) -> CellSignal:
        """Trigger immediate garbage collection."""
        before = len(self._garbage_queue)
        await self._collect_garbage(force=True)
        after = len(self._garbage_queue)
        
        return CellSignal(
            signal_type="gc_complete",
            source_cell=self.config.cell_id,
            target_cell=signal.source_cell,
            payload={
                "collected": before - after,
                "remaining": after,
                "total_collected": self._collected_garbage,
            },
        )
    
    async def _collect_garbage(self, force: bool = False):
        """Collect garbage items."""
        if not self._garbage_queue:
            return
        
        # Collect up to 10 items per heartbeat (or all if forced)
        collect_count = len(self._garbage_queue) if force else min(10, len(self._garbage_queue))
        
        for _ in range(collect_count):
            if not self._garbage_queue:
                break
            
            item = self._garbage_queue.pop(0)
            
            # Process based on type
            if item.item_type == "orphaned_route":
                self._flow_metrics.pop(item.item_id, None)
            elif item.item_type == "dead_signal":
                # Just log it
                logger.debug("GC: dead signal %s", item.item_id)
            elif item.item_type == "stale_cache":
                # Already handled by cache cleanup
                pass
            
            self._collected_garbage += 1
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HEALTH MONITORING
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _update_health(self):
        """Update health metrics."""
        # Mesh health based on connection health
        if self.connections:
            conn_health = sum(1 for c in self.connections.values() if c.websocket) / len(self.connections)
        else:
            conn_health = 0.5
        
        # Flow health based on queue backlog and drop rate
        total_queued = sum(q.qsize() for q in self._flow_queues.values())
        queue_health = max(0.0, 1.0 - (total_queued / 1000))  # Penalize large queues
        
        total_processed = sum(m.signals_processed for m in self._flow_metrics.values())
        total_dropped = sum(m.signals_dropped for m in self._flow_metrics.values())
        if total_processed + total_dropped > 0:
            success_rate = total_processed / (total_processed + total_dropped)
        else:
            success_rate = 1.0
        
        self._mesh_health = conn_health
        self._flow_health = (queue_health + success_rate) / 2


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Run Synapse cell standalone."""
    config = CellConfig.from_env()
    config.cell_type = CellType.SYNAPSE
    
    cell = SynapseCell(config)
    
    try:
        await cell.run_forever()
    except KeyboardInterrupt:
        await cell.shutdown()


if __name__ == "__main__":
    print("🔀 Starting Synapse Cell...")
    asyncio.run(main())

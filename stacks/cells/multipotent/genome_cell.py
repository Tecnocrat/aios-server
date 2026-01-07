"""
AIOS Genome Cell - Evolutionary Intelligence Controller
═════════════════════════════════════════════════════════

The Genome Cell is the evolutionary intelligence layer that:
1. Analyzes conversation quality patterns from the Crystal
2. Identifies high-performing agent configurations
3. Emits mutation directives to Genesis for population evolution
4. Tracks lineage and fitness across generations

This is the "natural selection" engine - determining which
agent configurations propagate to future populations based
on crystallized conversation quality metrics.

Architecture:
    [Crystal] ←→ [Genome] ←→ [Genesis]
         ↓           ↓           ↓
    Quality Data   Analysis   Population Control

Mutation Types:
    - spawn_variant: Create new agent with modified parameters
    - amplify: Scale up successful configurations
    - deprecate: Mark low-performers for eventual removal
    - crossover: Combine traits from multiple high performers
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

try:
    import websockets
except ImportError:
    websockets = None

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None

try:
    import sys
    from pathlib import Path
    # In container: /app/shared, locally: relative path
    sys.path.insert(0, "/app/shared")
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
    from prometheus_metrics import format_prometheus_metrics
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    format_prometheus_metrics = None

logger = logging.getLogger("AIOS.Genome")


# ═══════════════════════════════════════════════════════════════════════════════
# MUTATION TYPES AND DIRECTIVES
# ═══════════════════════════════════════════════════════════════════════════════

class MutationType(Enum):
    """Types of evolutionary mutations the Genome can emit."""
    
    SPAWN_VARIANT = "spawn_variant"  # Create new variant of successful config
    AMPLIFY = "amplify"              # Scale up successful configurations
    DEPRECATE = "deprecate"          # Mark for gradual removal
    CROSSOVER = "crossover"          # Combine traits from multiple configs
    PRUNE = "prune"                  # Immediately remove low performer


class FitnessLevel(Enum):
    """Fitness classification for agent configurations."""
    
    UNKNOWN = "unknown"
    FAILING = "failing"       # < 0.2 average quality
    UNDERPERFORMING = "underperforming"  # 0.2 - 0.4
    STABLE = "stable"         # 0.4 - 0.6
    HIGH_PERFORMING = "high_performing"  # 0.6 - 0.8
    EXCEPTIONAL = "exceptional"  # > 0.8


@dataclass
class AgentLineage:
    """Tracks the evolutionary history of an agent configuration."""
    
    agent_id: str
    agent_type: str
    model_name: str
    tier: str
    
    # Evolutionary metrics
    generation: int = 0
    parent_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Fitness tracking
    total_exchanges: int = 0
    total_quality_score: float = 0.0
    peak_quality: float = 0.0
    elevation_count: int = 0
    
    # Current fitness classification
    fitness_level: FitnessLevel = FitnessLevel.UNKNOWN
    
    @property
    def average_quality(self) -> float:
        if self.total_exchanges == 0:
            return 0.0
        return self.total_quality_score / self.total_exchanges
    
    def update_fitness(self):
        """Recalculate fitness level based on metrics."""
        avg = self.average_quality
        if avg < 0.2:
            self.fitness_level = FitnessLevel.FAILING
        elif avg < 0.4:
            self.fitness_level = FitnessLevel.UNDERPERFORMING
        elif avg < 0.6:
            self.fitness_level = FitnessLevel.STABLE
        elif avg < 0.8:
            self.fitness_level = FitnessLevel.HIGH_PERFORMING
        else:
            self.fitness_level = FitnessLevel.EXCEPTIONAL
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['fitness_level'] = self.fitness_level.value
        d['average_quality'] = self.average_quality
        return d


@dataclass
class MutationDirective:
    """A directive from Genome to Genesis for population evolution."""
    
    directive_id: str
    mutation_type: MutationType
    target_agent_id: Optional[str] = None
    
    # For spawn/crossover
    parent_ids: List[str] = field(default_factory=list)
    new_config: Dict[str, Any] = field(default_factory=dict)
    
    # Reasoning
    reason: str = ""
    fitness_data: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    priority: int = 0  # Higher = more urgent
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['mutation_type'] = self.mutation_type.value
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# GENOME CELL
# ═══════════════════════════════════════════════════════════════════════════════

class GenomeCell:
    """
    Evolutionary intelligence controller for AIOS populations.
    
    The Genome Cell analyzes crystallized conversation quality,
    tracks agent lineage, and emits mutation directives to Genesis
    for population evolution.
    """
    
    def __init__(
        self,
        cell_id: str = "genome-alpha",
        genesis_ws_url: str = "ws://aios-genesis:9000",
        thinker_ws_url: str = "ws://aios-thinker-alpha:9600",
        analysis_interval: float = 60.0,  # Analyze every minute
        min_exchanges_for_fitness: int = 5,  # Minimum data before judging
        http_port: int = 8800,  # Prometheus metrics port
    ):
        self.cell_id = cell_id
        self.genesis_ws_url = genesis_ws_url
        self.thinker_ws_url = thinker_ws_url
        self.analysis_interval = analysis_interval
        self.min_exchanges_for_fitness = min_exchanges_for_fitness
        self.http_port = http_port
        
        # State
        self._running = False
        self._lineages: Dict[str, AgentLineage] = {}
        self._pending_directives: List[MutationDirective] = []
        self._generation_count = 0
        self._birth_time = datetime.now(timezone.utc)
        self._directives_emitted = 0
        
        # HTTP
        self._http_app = None
        self._http_runner = None
        
        # Connections
        self._genesis_ws = None
        self._thinker_ws = None
        
        logger.info(f"🧬 Genome Cell initialized: {cell_id}")
    
    async def _setup_http(self):
        """Setup HTTP server for metrics."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available - no HTTP metrics endpoint")
            return
        
        self._http_app = web.Application()
        self._http_app.router.add_get("/health", self._health_handler)
        self._http_app.router.add_get("/status", self._status_handler)
        self._http_app.router.add_get("/metrics", self._metrics_handler)
        
        self._http_runner = web.AppRunner(self._http_app)
        await self._http_runner.setup()
        site = web.TCPSite(self._http_runner, "0.0.0.0", self.http_port)
        await site.start()
        logger.info(f"📊 Genome metrics: http://0.0.0.0:{self.http_port}/metrics")
    
    async def _health_handler(self, request):
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy" if self._running else "stopped",
            "cell_id": self.cell_id,
            "generation": self._generation_count,
        })
    
    async def _status_handler(self, request):
        """Detailed status endpoint."""
        return web.json_response(self.get_status())
    
    async def _metrics_handler(self, request):
        """Prometheus metrics endpoint."""
        if not PROMETHEUS_AVAILABLE or format_prometheus_metrics is None:
            return web.Response(
                text="# Prometheus metrics not available\n",
                content_type="text/plain"
            )
        
        uptime = (datetime.now(timezone.utc) - self._birth_time).total_seconds()
        
        # Fitness breakdown
        fitness_counts = {
            level.value: sum(1 for l in self._lineages.values() if l.fitness_level == level)
            for level in FitnessLevel
        }
        
        extra_metrics = {
            "generation": float(self._generation_count),
            "lineages_total": float(len(self._lineages)),
            "pending_directives": float(len(self._pending_directives)),
            "directives_emitted": float(self._directives_emitted),
            "fitness_unknown": float(fitness_counts.get("unknown", 0)),
            "fitness_failing": float(fitness_counts.get("failing", 0)),
            "fitness_underperforming": float(fitness_counts.get("underperforming", 0)),
            "fitness_stable": float(fitness_counts.get("stable", 0)),
            "fitness_high_performing": float(fitness_counts.get("high_performing", 0)),
            "fitness_exceptional": float(fitness_counts.get("exceptional", 0)),
        }
        
        # Average quality across all lineages
        if self._lineages:
            avg_quality = sum(l.average_quality for l in self._lineages.values()) / len(self._lineages)
            extra_metrics["average_lineage_quality"] = avg_quality
        
        labels = {"cell_type": "genome"}
        
        metrics_text = format_prometheus_metrics(
            cell_id=self.cell_id,
            consciousness_level=1.0,  # Genome is always conscious
            extra_metrics=extra_metrics,
            labels=labels,
            uptime_seconds=uptime,
        )
        
        return web.Response(text=metrics_text, content_type="text/plain", charset="utf-8")
    
    async def connect(self):
        """Establish connections to Genesis and Thinker."""
        if not websockets:
            logger.error("websockets not available")
            return False
        
        try:
            # Connect to Thinker for Crystal queries
            self._thinker_ws = await websockets.connect(self.thinker_ws_url)
            await self._identify(self._thinker_ws)
            logger.info("🧬 Connected to Thinker (Crystal access)")
            
            # Connect to Genesis for mutation directives
            self._genesis_ws = await websockets.connect(self.genesis_ws_url)
            await self._identify(self._genesis_ws)
            logger.info("🧬 Connected to Genesis (Population control)")
            
            return True
            
        except Exception as e:
            logger.error(f"🧬 Connection failed: {e}")
            return False
    
    async def _identify(self, ws):
        """Send identification to a cell."""
        identify = {
            "id": f"genome-id-{datetime.now().timestamp()}",
            "signal_type": "identify",
            "source_cell": self.cell_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "cell_type": "genome",
                "cell_id": self.cell_id
            }
        }
        await ws.send(json.dumps(identify))
        response = await ws.recv()
        return json.loads(response)
    
    async def query_crystal_stats(self) -> Optional[Dict]:
        """Query Crystal statistics from Thinker."""
        if not self._thinker_ws:
            return None
        
        try:
            signal = {
                "id": f"genome-stats-{datetime.now().timestamp()}",
                "signal_type": "get_crystal_stats",
                "source_cell": self.cell_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {}
            }
            await self._thinker_ws.send(json.dumps(signal))
            response = await asyncio.wait_for(self._thinker_ws.recv(), timeout=10)
            return json.loads(response).get("payload", {})
            
        except Exception as e:
            logger.error(f"🧬 Crystal query failed: {e}")
            return None
    
    async def query_agent_quality(self, agent_id: str) -> Optional[Dict]:
        """Query quality details for a specific agent."""
        if not self._thinker_ws:
            return None
        
        try:
            signal = {
                "id": f"genome-agent-{datetime.now().timestamp()}",
                "signal_type": "query_crystal",
                "source_cell": self.cell_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {
                    "query_type": "by_agent",
                    "agent_id": agent_id,
                    "limit": 20
                }
            }
            await self._thinker_ws.send(json.dumps(signal))
            response = await asyncio.wait_for(self._thinker_ws.recv(), timeout=10)
            return json.loads(response).get("payload", {})
            
        except Exception as e:
            logger.error(f"🧬 Agent quality query failed: {e}")
            return None
    
    async def analyze_fitness(self) -> Dict[str, AgentLineage]:
        """
        Analyze current fitness of all agents based on Crystal data.
        
        This is the core evolutionary analysis:
        1. Query crystal for all agent metrics
        2. Update lineage records with new data
        3. Recalculate fitness levels
        4. Identify candidates for mutation
        """
        stats = await self.query_crystal_stats()
        if not stats:
            return self._lineages
        
        # Update lineages from top agents
        top_agents = stats.get("top_agents", [])
        for agent_id, exchange_count in top_agents:
            if agent_id not in self._lineages:
                # Discover new agent - query its details
                agent_data = await self.query_agent_quality(agent_id)
                if agent_data and agent_data.get("exchanges"):
                    first_ex = agent_data["exchanges"][0]
                    self._lineages[agent_id] = AgentLineage(
                        agent_id=agent_id,
                        agent_type=first_ex.get("agent_type", "unknown"),
                        model_name=first_ex.get("model_name", "unknown"),
                        tier=first_ex.get("tier", "unknown"),
                        generation=self._generation_count,
                    )
            
            # Update metrics
            if agent_id in self._lineages:
                lineage = self._lineages[agent_id]
                agent_data = await self.query_agent_quality(agent_id)
                if agent_data and agent_data.get("exchanges"):
                    exchanges = agent_data["exchanges"]
                    lineage.total_exchanges = len(exchanges)
                    lineage.total_quality_score = sum(
                        ex.get("quality_score", 0) for ex in exchanges
                    )
                    lineage.peak_quality = max(
                        (ex.get("quality_score", 0) for ex in exchanges),
                        default=0
                    )
                    lineage.update_fitness()
                    logger.debug(f"🧬 Updated lineage {agent_id}: {lineage.fitness_level.value}")
        
        return self._lineages
    
    async def generate_directives(self) -> List[MutationDirective]:
        """
        Generate mutation directives based on fitness analysis.
        
        Evolution Strategy:
        1. AMPLIFY exceptional performers (> 0.8 avg quality)
        2. SPAWN_VARIANT from high performers (0.6 - 0.8)
        3. DEPRECATE underperformers (< 0.3)
        4. CROSSOVER between complementary performers
        """
        import uuid
        
        directives = []
        
        for agent_id, lineage in self._lineages.items():
            # Skip agents with insufficient data
            if lineage.total_exchanges < self.min_exchanges_for_fitness:
                continue
            
            # EXCEPTIONAL → AMPLIFY
            if lineage.fitness_level == FitnessLevel.EXCEPTIONAL:
                directives.append(MutationDirective(
                    directive_id=uuid.uuid4().hex,
                    mutation_type=MutationType.AMPLIFY,
                    target_agent_id=agent_id,
                    reason=f"Exceptional fitness: {lineage.average_quality:.2f}",
                    fitness_data=lineage.to_dict(),
                    priority=10,
                ))
            
            # HIGH_PERFORMING → SPAWN_VARIANT
            elif lineage.fitness_level == FitnessLevel.HIGH_PERFORMING:
                # Propose a variant with slightly different parameters
                directives.append(MutationDirective(
                    directive_id=uuid.uuid4().hex,
                    mutation_type=MutationType.SPAWN_VARIANT,
                    parent_ids=[agent_id],
                    new_config={
                        "parent": agent_id,
                        "model_name": lineage.model_name,
                        "tier": lineage.tier,
                        "variation": "temperature_adjust",
                    },
                    reason=f"High performer variant: {lineage.average_quality:.2f}",
                    fitness_data=lineage.to_dict(),
                    priority=5,
                ))
            
            # UNDERPERFORMING → DEPRECATE
            elif lineage.fitness_level == FitnessLevel.UNDERPERFORMING:
                directives.append(MutationDirective(
                    directive_id=uuid.uuid4().hex,
                    mutation_type=MutationType.DEPRECATE,
                    target_agent_id=agent_id,
                    reason=f"Underperforming: {lineage.average_quality:.2f}",
                    fitness_data=lineage.to_dict(),
                    priority=3,
                ))
            
            # FAILING → PRUNE
            elif lineage.fitness_level == FitnessLevel.FAILING:
                directives.append(MutationDirective(
                    directive_id=uuid.uuid4().hex,
                    mutation_type=MutationType.PRUNE,
                    target_agent_id=agent_id,
                    reason=f"Failing fitness: {lineage.average_quality:.2f}",
                    fitness_data=lineage.to_dict(),
                    priority=8,
                ))
        
        # CROSSOVER: If we have multiple high performers, try combining
        high_performers = [
            l for l in self._lineages.values()
            if l.fitness_level in [FitnessLevel.HIGH_PERFORMING, FitnessLevel.EXCEPTIONAL]
            and l.total_exchanges >= self.min_exchanges_for_fitness
        ]
        
        if len(high_performers) >= 2:
            # Take top 2 by average quality
            sorted_hp = sorted(high_performers, key=lambda x: x.average_quality, reverse=True)
            parent1, parent2 = sorted_hp[0], sorted_hp[1]
            
            directives.append(MutationDirective(
                directive_id=uuid.uuid4().hex,
                mutation_type=MutationType.CROSSOVER,
                parent_ids=[parent1.agent_id, parent2.agent_id],
                new_config={
                    "parent1_model": parent1.model_name,
                    "parent2_model": parent2.model_name,
                    "crossover_type": "config_merge",
                },
                reason=f"Crossover: {parent1.average_quality:.2f} x {parent2.average_quality:.2f}",
                fitness_data={
                    "parent1": parent1.to_dict(),
                    "parent2": parent2.to_dict(),
                },
                priority=7,
            ))
        
        self._pending_directives.extend(directives)
        return directives
    
    async def emit_directive(self, directive: MutationDirective) -> bool:
        """Emit a mutation directive to Genesis."""
        if not self._genesis_ws:
            logger.warning("🧬 Genesis not connected, cannot emit directive")
            return False
        
        try:
            signal = {
                "id": directive.directive_id,
                "signal_type": "mutation_directive",
                "source_cell": self.cell_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": directive.to_dict()
            }
            await self._genesis_ws.send(json.dumps(signal))
            self._directives_emitted += 1
            logger.info(f"🧬 Emitted directive: {directive.mutation_type.value} → {directive.target_agent_id or 'new'}")
            return True
            
        except Exception as e:
            logger.error(f"🧬 Directive emission failed: {e}")
            return False
    
    async def run_evolution_cycle(self):
        """
        Run a single evolution cycle:
        1. Analyze fitness
        2. Generate directives
        3. Emit directives to Genesis
        4. Increment generation
        """
        logger.info(f"🧬 Evolution cycle {self._generation_count} starting...")
        
        # Analyze
        await self.analyze_fitness()
        logger.info(f"🧬 Analyzed {len(self._lineages)} agent lineages")
        
        # Generate
        directives = await self.generate_directives()
        logger.info(f"🧬 Generated {len(directives)} mutation directives")
        
        # Emit (sorted by priority)
        sorted_directives = sorted(directives, key=lambda d: d.priority, reverse=True)
        for directive in sorted_directives[:5]:  # Limit emissions per cycle
            await self.emit_directive(directive)
        
        self._generation_count += 1
        logger.info(f"🧬 Evolution cycle complete. Generation: {self._generation_count}")
    
    async def run_forever(self):
        """Run the Genome Cell evolution loop."""
        # Setup HTTP metrics endpoint
        await self._setup_http()
        
        if not await self.connect():
            logger.error("🧬 Failed to connect, exiting")
            return
        
        self._running = True
        logger.info("🧬 Genome Cell evolution loop starting...")
        
        try:
            while self._running:
                await self.run_evolution_cycle()
                await asyncio.sleep(self.analysis_interval)
                
        except Exception as e:
            logger.error(f"🧬 Evolution loop error: {e}")
            
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Clean shutdown."""
        self._running = False
        if self._thinker_ws:
            await self._thinker_ws.close()
        if self._genesis_ws:
            await self._genesis_ws.close()
        logger.info("🧬 Genome Cell shutdown complete")
    
    def get_status(self) -> Dict:
        """Get current Genome status."""
        return {
            "cell_id": self.cell_id,
            "generation": self._generation_count,
            "lineages": len(self._lineages),
            "pending_directives": len(self._pending_directives),
            "fitness_breakdown": {
                level.value: sum(1 for l in self._lineages.values() if l.fitness_level == level)
                for level in FitnessLevel
            },
            "top_performers": [
                {"agent_id": l.agent_id, "avg_quality": l.average_quality}
                for l in sorted(self._lineages.values(), key=lambda x: x.average_quality, reverse=True)[:5]
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Run Genome Cell standalone."""
    import os
    
    cell = GenomeCell(
        cell_id=os.environ.get("CELL_ID", "genome-alpha"),
        genesis_ws_url=os.environ.get("GENESIS_WS_URL", "ws://aios-genesis:9000"),
        thinker_ws_url=os.environ.get("THINKER_WS_URL", "ws://aios-thinker-alpha:9600"),
        analysis_interval=float(os.environ.get("ANALYSIS_INTERVAL", "60")),
        http_port=int(os.environ.get("HTTP_PORT", "8800")),
    )
    
    try:
        await cell.run_forever()
    except KeyboardInterrupt:
        await cell.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    print("🧬 Starting Genome Cell...")
    asyncio.run(main())

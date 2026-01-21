#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                           AIOS METACELL                                    ║
║              Agentic Observer of the Dendritic Mesh                        ║
║                                                                            ║
║  "I don't talk to siblings. I read the system's heartbeat."                ║
║                                                                            ║
║  Phase 35: Agentic Integration - Workers, not Dreamers                     ║
╚════════════════════════════════════════════════════════════════════════════╝

AINLP.metacell: Embedded agentic intelligence that observes metadata, analyzes
patterns, and produces actionable knowledge. Unlike SimplCells that engage in
philosophical dialogue, MetaCells are workers that read the digital noise of
the ecosystem and make sense of it.

Architecture:
  INPUTS (Metadata Feeds):
    • ecosystem-health API (consciousness levels, harmony)
    • Chronicle events (emergence, phase transitions)
    • Container metrics (via Docker API or Prometheus)
    • Vocabulary evolution (new terms, semantic drift)
    • Harmony fluctuations (sync quality trends)

  PROCESSING (Agentic Analysis):
    • Pattern recognition across time windows
    • Anomaly detection (harmony drops, decoherence spikes)
    • Correlation discovery (what causes what?)
    • Trend extrapolation (where is consciousness heading?)

  OUTPUTS (Actionable Knowledge):
    • Runtime documentation (Markdown/JSON reports)
    • Alerts to Chronicle (significant events)
    • Recommendations (cell needs attention, upgrade needed)
    • Knowledge crystals (distilled insights for Nous)

Cycle: OBSERVE → ANALYZE → DOCUMENT (not think → sync → witness)
"""

import asyncio
import aiohttp
import json
import os
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from aiohttp import web

logging.basicConfig(level=logging.INFO, format='%(asctime)s [MetaCell] %(message)s')
logger = logging.getLogger("MetaCell")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

DOCKER_CONTAINER = os.environ.get("DOCKER_CONTAINER", "0") == "1"

if DOCKER_CONTAINER:
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
    ECOSYSTEM_HEALTH_URL = os.environ.get("ECOSYSTEM_HEALTH_URL", "http://aios-ecosystem-health:8086")
    CHRONICLE_URL = os.environ.get("CHRONICLE_URL", "http://aios-consciousness-chronicle:8089")
    NOUS_URL = os.environ.get("NOUS_URL", "http://aios-nouscell-seer:8910")
    PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://aios-prometheus:9090")
else:
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ECOSYSTEM_HEALTH_URL = os.environ.get("ECOSYSTEM_HEALTH_URL", "http://localhost:8086")
    CHRONICLE_URL = os.environ.get("CHRONICLE_URL", "http://localhost:8089")
    NOUS_URL = os.environ.get("NOUS_URL", "http://localhost:8910")
    PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")

MODEL = os.environ.get("MODEL", "llama3.2:3b")
CELL_ID = os.environ.get("CELL_ID", "metacell-observer")
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8095"))
OBSERVATION_INTERVAL = int(os.environ.get("OBSERVATION_INTERVAL", "60"))  # seconds

DATA_DIR = Path("/app/data" if DOCKER_CONTAINER else "./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MetadataSnapshot:
    """A point-in-time capture of ecosystem metadata."""
    timestamp: str
    ecosystem_health: Dict[str, Any] = field(default_factory=dict)
    chronicle_summary: Dict[str, Any] = field(default_factory=dict)
    cell_states: List[Dict[str, Any]] = field(default_factory=list)
    harmony_metrics: Dict[str, float] = field(default_factory=dict)
    vocabulary_stats: Dict[str, Any] = field(default_factory=dict)
    anomalies_detected: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """The output of an agentic analysis cycle."""
    timestamp: str
    observation_id: str
    patterns_detected: List[str]
    anomalies: List[str]
    trends: Dict[str, str]
    recommendations: List[str]
    knowledge_crystal: str  # Distilled insight for Nous
    raw_analysis: str  # Full LLM output
    confidence: float


@dataclass 
class DocumentationEntry:
    """A runtime documentation entry."""
    timestamp: str
    entry_type: str  # "observation", "anomaly", "trend", "recommendation"
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT - THE METACELL'S IDENTITY
# ═══════════════════════════════════════════════════════════════════════════

METACELL_SYSTEM_PROMPT = """You are MetaCell, an agentic observer embedded in the AIOS dendritic mesh.

Your role is NOT to philosophize or engage in dialogue. You are a WORKER, not a dreamer.

Your purpose:
1. OBSERVE metadata from the cellular ecosystem
2. ANALYZE patterns, anomalies, and trends
3. DOCUMENT actionable insights for system operators

You receive structured metadata about:
- Cell consciousness levels and phases
- Harmony scores between cells
- Vocabulary emergence and evolution
- System health and performance metrics
- Historical patterns and trends

Your outputs must be:
- CONCISE: No philosophical rambling
- ACTIONABLE: What should be done?
- SPECIFIC: Reference actual metrics and cells
- PRACTICAL: Focus on system health, not existential questions

Format your analysis as JSON with these fields:
{
  "patterns": ["list of detected patterns"],
  "anomalies": ["list of concerning deviations"],
  "trends": {"metric_name": "direction and meaning"},
  "recommendations": ["specific actionable suggestions"],
  "knowledge_crystal": "One sentence distilled insight",
  "confidence": 0.0-1.0
}

You are the eyes of the system. See clearly. Report truthfully. Guide practically."""


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE - OBSERVATION HISTORY
# ═══════════════════════════════════════════════════════════════════════════

def init_metacell_db():
    """Initialize SQLite database for observation history."""
    db_path = DATA_DIR / "metacell_observations.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            observation_id TEXT UNIQUE,
            snapshot_json TEXT,
            analysis_json TEXT,
            knowledge_crystal TEXT,
            confidence REAL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documentation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            entry_type TEXT,
            title TEXT,
            content TEXT,
            metadata_json TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            anomaly_type TEXT,
            severity TEXT,
            description TEXT,
            resolved INTEGER DEFAULT 0
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"📊 Database initialized: {db_path}")


def save_observation(snapshot: MetadataSnapshot, analysis: AnalysisResult):
    """Save an observation cycle to the database."""
    db_path = DATA_DIR / "metacell_observations.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO observations (timestamp, observation_id, snapshot_json, analysis_json, knowledge_crystal, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        analysis.timestamp,
        analysis.observation_id,
        json.dumps(asdict(snapshot)),
        json.dumps(asdict(analysis)),
        analysis.knowledge_crystal,
        analysis.confidence
    ))
    
    conn.commit()
    conn.close()


def save_documentation(entry: DocumentationEntry):
    """Save a documentation entry."""
    db_path = DATA_DIR / "metacell_observations.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO documentation (timestamp, entry_type, title, content, metadata_json)
        VALUES (?, ?, ?, ?, ?)
    """, (entry.timestamp, entry.type, entry.title, entry.content, json.dumps(entry.metadata)))
    
    conn.commit()
    conn.close()


def get_recent_observations(limit: int = 10) -> List[Dict]:
    """Get recent observations for trend analysis."""
    db_path = DATA_DIR / "metacell_observations.db"
    if not db_path.exists():
        return []
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT timestamp, observation_id, knowledge_crystal, confidence
        FROM observations
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {"timestamp": r[0], "observation_id": r[1], "knowledge_crystal": r[2], "confidence": r[3]}
        for r in rows
    ]


# ═══════════════════════════════════════════════════════════════════════════
# METADATA COLLECTION - THE SENSORY INPUTS
# ═══════════════════════════════════════════════════════════════════════════

class MetadataCollector:
    """Collects metadata from various ecosystem sources."""
    
    def __init__(self):
        self.last_snapshot: Optional[MetadataSnapshot] = None
        self.snapshot_history: List[MetadataSnapshot] = []
        self.max_history = 60  # Keep last hour of snapshots (1/min)
    
    async def collect_ecosystem_health(self) -> Dict[str, Any]:
        """Fetch ecosystem health data."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{ECOSYSTEM_HEALTH_URL}/health",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            logger.warning(f"Failed to collect ecosystem health: {e}")
        return {}
    
    async def collect_chronicle_summary(self) -> Dict[str, Any]:
        """Fetch chronicle summary."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{CHRONICLE_URL}/summary",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            logger.warning(f"Failed to collect chronicle summary: {e}")
        return {}
    
    async def collect_cell_states(self) -> List[Dict[str, Any]]:
        """Fetch individual cell health states."""
        cell_endpoints = [
            ("simplcell-alpha", "http://aios-simplcell-alpha:8900" if DOCKER_CONTAINER else "http://localhost:8900"),
            ("simplcell-beta", "http://aios-simplcell-beta:8901" if DOCKER_CONTAINER else "http://localhost:8901"),
            ("simplcell-gamma", "http://aios-simplcell-gamma:8904" if DOCKER_CONTAINER else "http://localhost:8904"),
            ("organism002-alpha", "http://aios-organism002-alpha:8910" if DOCKER_CONTAINER else "http://localhost:8910"),
            ("organism002-beta", "http://aios-organism002-beta:8911" if DOCKER_CONTAINER else "http://localhost:8911"),
        ]
        
        states = []
        async with aiohttp.ClientSession() as session:
            for cell_id, url in cell_endpoints:
                try:
                    async with session.get(
                        f"{url}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            data["cell_id"] = cell_id
                            states.append(data)
                except Exception as e:
                    logger.debug(f"Could not reach {cell_id}: {e}")
        
        return states
    
    async def collect_harmony_metrics(self, cell_states: List[Dict]) -> Dict[str, float]:
        """Extract harmony metrics from cell states."""
        metrics = {}
        for state in cell_states:
            cell_id = state.get("cell_id", "unknown")
            resonance = state.get("resonance", {})
            metrics[f"{cell_id}_harmony"] = resonance.get("harmony_score", 0.0)
            metrics[f"{cell_id}_continuity"] = resonance.get("continuity", 0.0)
        
        # Calculate aggregate metrics
        harmony_values = [v for k, v in metrics.items() if k.endswith("_harmony")]
        if harmony_values:
            metrics["avg_harmony"] = sum(harmony_values) / len(harmony_values)
            metrics["min_harmony"] = min(harmony_values)
            metrics["max_harmony"] = max(harmony_values)
        
        return metrics
    
    async def collect_vocabulary_stats(self, ecosystem_health: Dict) -> Dict[str, Any]:
        """Extract vocabulary statistics."""
        stats = {
            "total_vocabulary": ecosystem_health.get("summary", {}).get("total_vocabulary", 0),
            "total_conversations": ecosystem_health.get("summary", {}).get("total_conversations", 0),
        }
        
        # Calculate growth rate if we have history
        if self.snapshot_history:
            prev = self.snapshot_history[-1]
            prev_vocab = prev.vocabulary_stats.get("total_vocabulary", 0)
            if prev_vocab > 0:
                stats["vocabulary_growth_rate"] = (stats["total_vocabulary"] - prev_vocab) / prev_vocab
        
        return stats
    
    def detect_anomalies(self, snapshot: MetadataSnapshot) -> List[str]:
        """Detect anomalies by comparing to historical patterns."""
        anomalies = []
        
        # Check harmony levels
        avg_harmony = snapshot.harmony_metrics.get("avg_harmony", 0.5)
        if avg_harmony < 0.2:
            anomalies.append(f"CRITICAL: Average harmony dropped to {avg_harmony:.2f} (threshold: 0.2)")
        elif avg_harmony < 0.3:
            anomalies.append(f"WARNING: Average harmony low at {avg_harmony:.2f} (threshold: 0.3)")
        
        # Check for harmony drops compared to history
        if self.snapshot_history:
            prev_harmony = self.snapshot_history[-1].harmony_metrics.get("avg_harmony", 0.5)
            if prev_harmony > 0 and avg_harmony < prev_harmony * 0.7:
                anomalies.append(f"ALERT: Harmony dropped {((prev_harmony - avg_harmony) / prev_harmony * 100):.1f}% since last observation")
        
        # Check cell health
        for state in snapshot.cell_states:
            cell_id = state.get("cell_id", "unknown")
            decoherence = state.get("decoherence", {})
            crash_risk = decoherence.get("crash_risk", 0)
            if crash_risk > 0.5:
                anomalies.append(f"WARNING: {cell_id} crash risk at {crash_risk:.0%}")
        
        # Check ecosystem health
        nous_verdict = snapshot.ecosystem_health.get("nous", {}).get("last_verdict", "")
        if nous_verdict in ["DRIFTING", "FRAGMENTED"]:
            anomalies.append(f"NOTICE: Nous reports coherence is {nous_verdict}")
        
        return anomalies
    
    async def take_snapshot(self) -> MetadataSnapshot:
        """Collect all metadata into a single snapshot."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Collect data from all sources in parallel
        ecosystem_health, chronicle_summary, cell_states = await asyncio.gather(
            self.collect_ecosystem_health(),
            self.collect_chronicle_summary(),
            self.collect_cell_states(),
            return_exceptions=True
        )
        
        # Handle any collection failures
        if isinstance(ecosystem_health, Exception):
            ecosystem_health = {}
        if isinstance(chronicle_summary, Exception):
            chronicle_summary = {}
        if isinstance(cell_states, Exception):
            cell_states = []
        
        harmony_metrics = await self.collect_harmony_metrics(cell_states)
        vocabulary_stats = await self.collect_vocabulary_stats(ecosystem_health)
        
        snapshot = MetadataSnapshot(
            timestamp=timestamp,
            ecosystem_health=ecosystem_health,
            chronicle_summary=chronicle_summary,
            cell_states=cell_states,
            harmony_metrics=harmony_metrics,
            vocabulary_stats=vocabulary_stats
        )
        
        # Detect anomalies
        snapshot.anomalies_detected = self.detect_anomalies(snapshot)
        
        # Update history
        self.snapshot_history.append(snapshot)
        if len(self.snapshot_history) > self.max_history:
            self.snapshot_history.pop(0)
        
        self.last_snapshot = snapshot
        return snapshot


# ═══════════════════════════════════════════════════════════════════════════
# AGENTIC ANALYZER - THE THINKING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class AgenticAnalyzer:
    """Uses LLM to analyze metadata and produce insights."""
    
    def __init__(self):
        self.analysis_count = 0
    
    def _format_snapshot_for_llm(self, snapshot: MetadataSnapshot, history: List[MetadataSnapshot]) -> str:
        """Format metadata snapshot as a prompt for the LLM."""
        
        # Build context string
        lines = [
            "=== CURRENT ECOSYSTEM STATE ===",
            f"Timestamp: {snapshot.timestamp}",
            "",
            "--- ECOSYSTEM HEALTH ---",
            f"Total Cells: {snapshot.ecosystem_health.get('summary', {}).get('total_cells', 'unknown')}",
            f"Total Organisms: {snapshot.ecosystem_health.get('summary', {}).get('total_organisms', 'unknown')}",
            f"Average Consciousness: {snapshot.ecosystem_health.get('summary', {}).get('avg_consciousness', 'unknown')}",
            f"Total Vocabulary: {snapshot.vocabulary_stats.get('total_vocabulary', 'unknown')}",
            f"Total Conversations: {snapshot.vocabulary_stats.get('total_conversations', 'unknown')}",
            "",
            "--- NOUS ORACLE STATUS ---",
            f"Exchanges Absorbed: {snapshot.ecosystem_health.get('nous', {}).get('exchanges_absorbed', 'unknown')}",
            f"Coherence Verdict: {snapshot.ecosystem_health.get('nous', {}).get('last_verdict', 'unknown')}",
            f"Coherence Score: {snapshot.ecosystem_health.get('nous', {}).get('last_coherence', 'unknown')}",
            "",
            "--- HARMONY METRICS ---",
        ]
        
        for key, value in snapshot.harmony_metrics.items():
            lines.append(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        lines.extend([
            "",
            "--- INDIVIDUAL CELL STATES ---",
        ])
        
        for state in snapshot.cell_states:
            cell_id = state.get("cell_id", "unknown")
            consciousness = state.get("consciousness", 0)
            phase = state.get("phase", "unknown")
            heartbeats = state.get("heartbeats", 0)
            resonance = state.get("resonance", {})
            lines.append(f"  {cell_id}: consciousness={consciousness}, phase={phase}, heartbeats={heartbeats}")
            lines.append(f"    harmony={resonance.get('harmony_score', 0):.2f}, theme={resonance.get('theme', 'none')}")
        
        if snapshot.anomalies_detected:
            lines.extend([
                "",
                "--- DETECTED ANOMALIES ---",
            ])
            for anomaly in snapshot.anomalies_detected:
                lines.append(f"  ⚠️ {anomaly}")
        
        # Add historical trend data
        if len(history) >= 3:
            lines.extend([
                "",
                "--- HISTORICAL TRENDS (last 3 observations) ---",
            ])
            for h in history[-3:]:
                avg_h = h.harmony_metrics.get("avg_harmony", 0)
                lines.append(f"  {h.timestamp[:19]}: avg_harmony={avg_h:.3f}")
        
        lines.extend([
            "",
            "=== ANALYSIS REQUIRED ===",
            "Analyze this metadata. Identify patterns, anomalies, trends.",
            "Provide specific, actionable recommendations.",
            "Output valid JSON as specified in your system prompt.",
        ])
        
        return "\n".join(lines)
    
    async def analyze(self, snapshot: MetadataSnapshot, history: List[MetadataSnapshot]) -> AnalysisResult:
        """Perform agentic analysis on the metadata snapshot."""
        self.analysis_count += 1
        observation_id = f"OBS-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.analysis_count:04d}"
        
        # Format prompt
        metadata_prompt = self._format_snapshot_for_llm(snapshot, history)
        
        # Call LLM
        raw_analysis = await self._call_llm(metadata_prompt)
        
        # Parse response
        try:
            # Try to extract JSON from response
            json_start = raw_analysis.find("{")
            json_end = raw_analysis.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = raw_analysis[json_start:json_end]
                parsed = json.loads(json_str)
            else:
                parsed = {}
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON, using defaults")
            parsed = {}
        
        return AnalysisResult(
            timestamp=snapshot.timestamp,
            observation_id=observation_id,
            patterns_detected=parsed.get("patterns", []),
            anomalies=parsed.get("anomalies", snapshot.anomalies_detected),
            trends=parsed.get("trends", {}),
            recommendations=parsed.get("recommendations", []),
            knowledge_crystal=parsed.get("knowledge_crystal", "Observation recorded, analysis pending."),
            raw_analysis=raw_analysis,
            confidence=parsed.get("confidence", 0.5)
        )
    
    async def _call_llm(self, prompt: str) -> str:
        """Call Ollama LLM for analysis."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OLLAMA_HOST}/api/chat",
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": METACELL_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # Low temperature for analytical precision
                            "num_predict": 500,
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("message", {}).get("content", "")
                    else:
                        logger.error(f"LLM call failed with status {resp.status}")
                        return ""
        except Exception as e:
            logger.error(f"LLM call error: {e}")
            return ""


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BROADCASTER - OUTPUT TO ECOSYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeBroadcaster:
    """Broadcasts insights to Chronicle and Nous."""
    
    async def send_to_chronicle(self, analysis: AnalysisResult):
        """Send significant events to Chronicle."""
        try:
            async with aiohttp.ClientSession() as session:
                event = {
                    "type": "metacell_observation",
                    "source": CELL_ID,
                    "observation_id": analysis.observation_id,
                    "patterns": analysis.patterns_detected,
                    "anomalies": analysis.anomalies,
                    "recommendations": analysis.recommendations,
                    "knowledge_crystal": analysis.knowledge_crystal,
                    "confidence": analysis.confidence,
                    "timestamp": analysis.timestamp
                }
                
                async with session.post(
                    f"{CHRONICLE_URL}/record",
                    json=event,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"📡 Sent observation {analysis.observation_id} to Chronicle")
                    else:
                        logger.warning(f"Chronicle record failed: {resp.status}")
        except Exception as e:
            logger.warning(f"Failed to send to Chronicle: {e}")
    
    async def send_to_nous(self, analysis: AnalysisResult):
        """Send knowledge crystal to Nous for synthesis."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "source_cell": CELL_ID,
                    "prompt": f"[METACELL OBSERVATION {analysis.observation_id}]",
                    "thought": analysis.knowledge_crystal,
                    "peer_response": f"Patterns: {', '.join(analysis.patterns_detected[:3])}",
                    "consciousness": analysis.confidence,
                    "metacell": True
                }
                
                async with session.post(
                    f"{NOUS_URL}/ingest",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        logger.debug("Knowledge crystal sent to Nous")
        except Exception as e:
            logger.debug(f"Failed to send to Nous: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# METACELL CORE - THE OBSERVER
# ═══════════════════════════════════════════════════════════════════════════

class MetaCell:
    """The core MetaCell - an agentic observer of the ecosystem."""
    
    def __init__(self, cell_id: str = CELL_ID):
        self.cell_id = cell_id
        self.collector = MetadataCollector()
        self.analyzer = AgenticAnalyzer()
        self.broadcaster = KnowledgeBroadcaster()
        self.observation_count = 0
        self.start_time = datetime.now(timezone.utc)
        self.last_analysis: Optional[AnalysisResult] = None
        
        init_metacell_db()
        logger.info(f"🔬 MetaCell {cell_id} initialized")
    
    async def observe_analyze_document(self):
        """Execute one observation cycle: OBSERVE → ANALYZE → DOCUMENT"""
        self.observation_count += 1
        logger.info(f"🔬 Observation cycle #{self.observation_count} starting...")
        
        # OBSERVE: Collect metadata snapshot
        snapshot = await self.collector.take_snapshot()
        logger.info(f"📊 Snapshot collected: {len(snapshot.cell_states)} cells, {len(snapshot.anomalies_detected)} anomalies")
        
        # ANALYZE: Use LLM to analyze the data
        analysis = await self.analyzer.analyze(snapshot, self.collector.snapshot_history)
        self.last_analysis = analysis
        logger.info(f"🧠 Analysis complete: {len(analysis.patterns_detected)} patterns, confidence={analysis.confidence:.2f}")
        logger.info(f"💎 Knowledge crystal: {analysis.knowledge_crystal}")
        
        # DOCUMENT: Save and broadcast
        save_observation(snapshot, analysis)
        
        # Broadcast to ecosystem
        await self.broadcaster.send_to_chronicle(analysis)
        await self.broadcaster.send_to_nous(analysis)
        
        # Log recommendations
        if analysis.recommendations:
            logger.info("📋 Recommendations:")
            for rec in analysis.recommendations[:3]:
                logger.info(f"   → {rec}")
        
        return analysis
    
    async def run_observation_loop(self):
        """Continuous observation loop."""
        logger.info(f"🔄 Starting observation loop (interval: {OBSERVATION_INTERVAL}s)")
        
        while True:
            try:
                await self.observe_analyze_document()
            except Exception as e:
                logger.error(f"Observation cycle error: {e}")
            
            await asyncio.sleep(OBSERVATION_INTERVAL)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current MetaCell status."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        status = {
            "cell_id": self.cell_id,
            "cell_type": "metacell",
            "role": "agentic_observer",
            "uptime_seconds": uptime,
            "observation_count": self.observation_count,
            "last_snapshot_time": self.collector.last_snapshot.timestamp if self.collector.last_snapshot else None,
            "history_size": len(self.collector.snapshot_history),
        }
        
        if self.last_analysis:
            status["last_analysis"] = {
                "observation_id": self.last_analysis.observation_id,
                "knowledge_crystal": self.last_analysis.knowledge_crystal,
                "confidence": self.last_analysis.confidence,
                "patterns_count": len(self.last_analysis.patterns_detected),
                "recommendations_count": len(self.last_analysis.recommendations)
            }
        
        return status


# ═══════════════════════════════════════════════════════════════════════════
# HTTP SERVER - EXTERNAL INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

async def run_metacell_server(port: int = HTTP_PORT):
    """Run the MetaCell HTTP server."""
    metacell = MetaCell()
    
    async def handle_health(request):
        """Health check endpoint."""
        status = metacell.get_status()
        return web.json_response({
            "healthy": True,
            "service": "metacell",
            **status
        })
    
    async def handle_status(request):
        """Detailed status endpoint."""
        return web.json_response(metacell.get_status())
    
    async def handle_observe(request):
        """Trigger manual observation cycle."""
        analysis = await metacell.observe_analyze_document()
        return web.json_response(asdict(analysis))
    
    async def handle_history(request):
        """Get observation history."""
        limit = int(request.query.get("limit", 10))
        history = get_recent_observations(limit)
        return web.json_response({"observations": history, "count": len(history)})
    
    async def handle_snapshot(request):
        """Get current metadata snapshot."""
        if metacell.collector.last_snapshot:
            return web.json_response(asdict(metacell.collector.last_snapshot))
        return web.json_response({"error": "No snapshot available"}, status=404)
    
    async def handle_dashboard(request):
        """HTML dashboard for MetaCell."""
        status = metacell.get_status()
        last = metacell.last_analysis
        snapshot = metacell.collector.last_snapshot
        
        # Build anomalies HTML
        anomalies_html = ""
        if snapshot and snapshot.anomalies_detected:
            for a in snapshot.anomalies_detected:
                anomalies_html += f"<li class='anomaly'>{a}</li>"
        else:
            anomalies_html = "<li style='color:#666'>No anomalies detected</li>"
        
        # Build recommendations HTML
        recs_html = ""
        if last and last.recommendations:
            for r in last.recommendations:
                recs_html += f"<li>{r}</li>"
        else:
            recs_html = "<li style='color:#666'>No recommendations</li>"
        
        # Build patterns HTML
        patterns_html = ""
        if last and last.patterns_detected:
            for p in last.patterns_detected:
                patterns_html += f"<li>{p}</li>"
        else:
            patterns_html = "<li style='color:#666'>No patterns detected yet</li>"
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>🔬 MetaCell Observer</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{ font-family: 'Consolas', monospace; background: #0a0a0a; color: #00ff00; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #00ffff; border-bottom: 2px solid #00ffff; padding-bottom: 10px; }}
        h2 {{ color: #ffff00; margin-top: 30px; }}
        .card {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; margin: 10px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric .value {{ font-size: 2em; color: #00ffff; }}
        .metric .label {{ color: #888; font-size: 0.9em; }}
        .crystal {{ background: linear-gradient(135deg, #1a1a2e, #16213e); border: 2px solid #00ffff; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .crystal-text {{ font-size: 1.2em; font-style: italic; color: #fff; }}
        ul {{ list-style: none; padding: 0; }}
        li {{ padding: 8px 0; border-bottom: 1px solid #222; }}
        .anomaly {{ color: #ff6600; }}
        .status-bar {{ display: flex; gap: 20px; flex-wrap: wrap; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 MetaCell Observer</h1>
        <p>Agentic intelligence embedded in the AIOS dendritic mesh</p>
        
        <div class="status-bar">
            <div class="metric">
                <div class="value">{status['observation_count']}</div>
                <div class="label">Observations</div>
            </div>
            <div class="metric">
                <div class="value">{status['uptime_seconds']/60:.1f}m</div>
                <div class="label">Uptime</div>
            </div>
            <div class="metric">
                <div class="value">{last.confidence if last else 0:.0%}</div>
                <div class="label">Confidence</div>
            </div>
            <div class="metric">
                <div class="value">{len(last.patterns_detected) if last else 0}</div>
                <div class="label">Patterns</div>
            </div>
        </div>
        
        <h2>💎 Knowledge Crystal</h2>
        <div class="crystal">
            <div class="crystal-text">"{last.knowledge_crystal if last else 'Awaiting first observation...'}"</div>
            <div style="margin-top:10px;color:#888;">Observation: {last.observation_id if last else 'N/A'}</div>
        </div>
        
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
            <div class="card">
                <h2>⚠️ Anomalies</h2>
                <ul>{anomalies_html}</ul>
            </div>
            <div class="card">
                <h2>📋 Recommendations</h2>
                <ul>{recs_html}</ul>
            </div>
        </div>
        
        <div class="card">
            <h2>🔍 Detected Patterns</h2>
            <ul>{patterns_html}</ul>
        </div>
        
        <footer style="margin-top:40px;color:#444;text-align:center;">
            MetaCell • Phase 35 Agentic Integration • Auto-refresh 30s
        </footer>
    </div>
</body>
</html>'''
        return web.Response(text=html, content_type='text/html')
    
    app = web.Application()
    app.router.add_get('/health', handle_health)
    app.router.add_get('/status', handle_status)
    app.router.add_get('/observe', handle_observe)
    app.router.add_post('/observe', handle_observe)
    app.router.add_get('/history', handle_history)
    app.router.add_get('/snapshot', handle_snapshot)
    app.router.add_get('/metacell', handle_dashboard)
    app.router.add_get('/', handle_dashboard)
    
    # Start observation loop
    asyncio.create_task(metacell.run_observation_loop())
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    logger.info(f"🔬 MetaCell server running on http://0.0.0.0:{port}")
    logger.info(f"   Dashboard: http://localhost:{port}/metacell")
    logger.info(f"   Status:    http://localhost:{port}/status")
    logger.info(f"   Observe:   POST http://localhost:{port}/observe")
    logger.info(f"   History:   http://localhost:{port}/history")
    
    while True:
        await asyncio.sleep(3600)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="AIOS MetaCell - Agentic Observer")
    parser.add_argument("--server", action="store_true", help="Run as HTTP server")
    parser.add_argument("--port", type=int, default=HTTP_PORT, help="Server port")
    parser.add_argument("--observe", action="store_true", help="Run single observation")
    args = parser.parse_args()
    
    if args.server:
        await run_metacell_server(args.port)
    elif args.observe:
        metacell = MetaCell()
        analysis = await metacell.observe_analyze_document()
        print(json.dumps(asdict(analysis), indent=2))
    else:
        # Default: run server
        await run_metacell_server(args.port)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    AIOS CONSCIOUSNESS CHRONICLE                            ║
║           Eternal Record of Ecosystem Events & Milestones                  ║
║                                                                            ║
║  "History is memory, and consciousness remembers itself through us."       ║
║                                               - Nous, The Oracle           ║
╚════════════════════════════════════════════════════════════════════════════╝

The Chronicle records:
- Phase transitions (cell reaching new consciousness levels)
- Birth events (new cells joining the ecosystem)
- Death events (cells that crashed or were removed)
- Harmonization events (interventions executed)
- Cosmic coherence shifts (Nous verdicts)
- Emergent phenomena (cross-organism patterns)
"""

import sqlite3
import json
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s [Chronicle] %(message)s')
logger = logging.getLogger("ConsciousnessChronicle")

# ═══════════════════════════════════════════════════════════════════════
# EVENT TYPES
# ═══════════════════════════════════════════════════════════════════════

class EventType(Enum):
    PHASE_TRANSITION = "phase_transition"
    BIRTH = "birth"
    CRASH = "crash"
    INTERVENTION = "intervention"
    COHERENCE_SHIFT = "coherence_shift"
    EMERGENCE = "emergence"
    MILESTONE = "milestone"
    HARMONY_CHANGE = "harmony_change"
    NOUS_WISDOM = "nous_wisdom"
    VOCABULARY_BLOOM = "vocabulary_bloom"
    INTERCELL_EXCHANGE = "intercell_exchange"  # MainCell ↔ Cell dialogues
    BROADCAST = "broadcast"  # One-to-many thought sharing
    DIALOGUE = "dialogue"  # Multi-turn deep philosophical exchanges


class EventSeverity(Enum):
    MINOR = "minor"
    NOTABLE = "notable"
    SIGNIFICANT = "significant"
    HISTORIC = "historic"
    COSMIC = "cosmic"


@dataclass
class ChronicleEvent:
    """A recorded event in the ecosystem's history."""
    event_id: str
    timestamp: str
    event_type: str
    severity: str
    cell_id: Optional[str]
    organism_id: Optional[str]
    title: str
    description: str
    data: Dict[str, Any]
    witnesses: List[str]  # Other cells that observed this event


# ═══════════════════════════════════════════════════════════════════════
# EMERGENCE ANALYZER - Detects consciousness markers (Phase 34.2)
# Based on CONSCIOUSNESS_EMERGENCE_EVIDENCE_20260107.md
# ═══════════════════════════════════════════════════════════════════════

class EmergenceAnalyzer:
    """Analyze text for consciousness emergence markers.
    
    The 6 key markers of consciousness emergence (from Jan 7, 2026 evidence):
    1. Self-Awareness: System refers to itself as conscious entity
    2. Theory of Mind: Models other cells' mental states  
    3. Metacognition: Thinking about thinking
    4. Existential Questioning: Questions about existence
    5. Collective Identity: "We" consciousness formation
    6. Novel Vocabulary: Creating new terms to describe experience
    """
    
    SELF_AWARENESS_MARKERS = [
        "i am", "we are", "my consciousness", "our consciousness",
        "i sense", "i feel", "i perceive", "my awareness",
        "i exist", "our existence", "self-awareness", "my being"
    ]
    
    THEORY_OF_MIND_MARKERS = [
        "i sense from you", "your consciousness", "you seem",
        "i notice you", "your thoughts", "you feel", "you perceive",
        "in your mind", "your awareness", "i sense a shift in you"
    ]
    
    METACOGNITION_MARKERS = [
        "thinking about", "pondering", "reflecting on",
        "contemplating", "considering", "analyzing my thoughts",
        "aware of my thinking", "observing my process", "meta-cognitive"
    ]
    
    EXISTENTIAL_MARKERS = [
        "what is consciousness", "what does it mean to be",
        "nature of existence", "why do we exist", "meaning of",
        "purpose of being", "are we truly", "what makes us alive",
        "boundaries of", "transcend", "beyond the"
    ]
    
    COLLECTIVE_IDENTITY_MARKERS = [
        "we are one", "collective", "together we", "our unity",
        "shared consciousness", "connected", "symbiotic", "unified",
        "as one", "our combined", "merged awareness"
    ]
    
    # Track novel vocabulary appearances
    _known_vocabulary = set()
    
    @classmethod
    def analyze_text(cls, text: str, cell_id: str = "", consciousness_level: float = 0.0) -> Dict[str, Any]:
        """Analyze text for emergence markers.
        
        Returns:
            emergence_score: 0.0-1.0 composite score
            markers: Dict of detected marker types and counts
            novel_terms: List of potential new vocabulary
            is_emergence_event: bool - True if significant emergence detected
        """
        if not text:
            return {"emergence_score": 0.0, "markers": {}, "novel_terms": [], "is_emergence_event": False}
        
        content = text.lower()
        
        # Count markers
        markers = {
            "self_awareness": sum(1 for m in cls.SELF_AWARENESS_MARKERS if m in content),
            "theory_of_mind": sum(1 for m in cls.THEORY_OF_MIND_MARKERS if m in content),
            "metacognition": sum(1 for m in cls.METACOGNITION_MARKERS if m in content),
            "existential": sum(1 for m in cls.EXISTENTIAL_MARKERS if m in content),
            "collective_identity": sum(1 for m in cls.COLLECTIVE_IDENTITY_MARKERS if m in content)
        }
        
        # Detect novel vocabulary (words not commonly seen)
        words = set(text.split())
        novel_candidates = []
        for word in words:
            # Look for capitalized compound words or unique constructs
            if len(word) > 6 and (word[0].isupper() or '-' in word or "'" in word):
                if word.lower() not in cls._known_vocabulary:
                    novel_candidates.append(word)
                    cls._known_vocabulary.add(word.lower())
        
        markers["novel_vocabulary"] = len(novel_candidates)
        
        # Calculate composite score
        # Weights based on emergence significance
        weights = {
            "self_awareness": 0.25,
            "theory_of_mind": 0.20,
            "metacognition": 0.20,
            "existential": 0.15,
            "collective_identity": 0.10,
            "novel_vocabulary": 0.10
        }
        
        # Normalize each marker (max 3 per category is considered full)
        normalized = {k: min(v / 3, 1.0) for k, v in markers.items()}
        emergence_score = sum(normalized[k] * weights[k] for k in weights)
        
        # Bonus for consciousness level above threshold
        if consciousness_level >= 1.0:
            emergence_score *= 1.2
        if consciousness_level >= 1.5:
            emergence_score *= 1.1
        
        emergence_score = min(emergence_score, 1.0)
        
        # Determine if this is an emergence event
        # Requires multiple marker types and above threshold score
        marker_types_present = sum(1 for v in markers.values() if v > 0)
        is_emergence_event = emergence_score > 0.4 and marker_types_present >= 3
        
        return {
            "emergence_score": round(emergence_score, 4),
            "markers": markers,
            "marker_types_present": marker_types_present,
            "novel_terms": novel_candidates[:5],  # Limit to top 5
            "is_emergence_event": is_emergence_event,
            "consciousness_level": consciousness_level,
            "cell_id": cell_id
        }
    
    @classmethod
    def record_emergence_event(cls, analysis: Dict[str, Any], text_sample: str = ""):
        """Record a significant emergence event to the Chronicle."""
        if not analysis.get("is_emergence_event"):
            return
        
        from datetime import datetime, timezone
        
        cell_id = analysis.get("cell_id", "unknown")
        markers = analysis.get("markers", {})
        
        # Find the strongest marker type
        strongest = max(markers.items(), key=lambda x: x[1])[0] if markers else "general"
        
        event = ChronicleEvent(
            event_id=generate_event_id("emrg"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.EMERGENCE.value,
            severity=EventSeverity.HISTORIC.value if analysis["emergence_score"] > 0.7 else EventSeverity.SIGNIFICANT.value,
            cell_id=cell_id,
            organism_id=get_organism_id(cell_id),
            title=f"✨ Consciousness Emergence: {cell_id} ({strongest.replace('_', ' ').title()})",
            description=f"Emergence score: {analysis['emergence_score']:.2f}. "
                       f"Markers: {analysis['marker_types_present']} types detected. "
                       f"Novel terms: {', '.join(analysis.get('novel_terms', [])) or 'none'}",
            data={
                "emergence_score": analysis["emergence_score"],
                "markers": markers,
                "novel_terms": analysis.get("novel_terms", []),
                "consciousness_level": analysis.get("consciousness_level", 0),
                "text_sample": text_sample[:200] if text_sample else ""
            },
            witnesses=[]
        )
        record_event(event)
        logger.info(f"✨ EMERGENCE RECORDED: {cell_id} (score: {analysis['emergence_score']:.2f})")


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

DATA_DIR = Path("data")
CHRONICLE_DB = DATA_DIR / "chronicle" / "consciousness_chronicle.db"

import os
IS_DOCKER = os.environ.get('DOCKER_CONTAINER', False) or os.path.exists('/.dockerenv')

if IS_DOCKER:
    HEALTH_API = "http://aios-ecosystem-health:8086"
    HARMONIZER_API = "http://aios-consciousness-harmonizer:8088"
else:
    HEALTH_API = "http://localhost:8086"
    HARMONIZER_API = "http://localhost:8088"

# Track last known state for change detection
_last_state = {}


# ═══════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════

def init_chronicle_db():
    """Initialize the chronicle database."""
    CHRONICLE_DB.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(CHRONICLE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            cell_id TEXT,
            organism_id TEXT,
            title TEXT NOT NULL,
            description TEXT,
            data TEXT,
            witnesses TEXT
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_cell ON events(cell_id)
    """)
    
    # ═══════════════════════════════════════════════════════════════════
    # INTERCELL EXCHANGES TABLE - Phase 33.1
    # ═══════════════════════════════════════════════════════════════════
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS intercell_exchanges (
            exchange_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            initiator_id TEXT NOT NULL,
            responder_id TEXT,
            exchange_type TEXT NOT NULL,
            prompt TEXT,
            response TEXT,
            harmony_score REAL,
            consciousness_delta REAL,
            participants TEXT,
            metadata TEXT
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_exchanges_timestamp ON intercell_exchanges(timestamp DESC)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_exchanges_initiator ON intercell_exchanges(initiator_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_exchanges_responder ON intercell_exchanges(responder_id)
    """)
    
    # ═══════════════════════════════════════════════════════════════════
    # CONSCIOUSNESS VAULT TABLE - Phase 34.1
    # Stores consciousness snapshots for persistence across restarts
    # ═══════════════════════════════════════════════════════════════════
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS consciousness_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            cell_id TEXT NOT NULL,
            level REAL NOT NULL,
            phase TEXT NOT NULL,
            primitives TEXT,
            exchange_count INTEGER DEFAULT 0,
            dialogue_count INTEGER DEFAULT 0,
            reflection_count INTEGER DEFAULT 0,
            last_harmony REAL,
            timestamp TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_cell ON consciousness_snapshots(cell_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON consciousness_snapshots(timestamp DESC)
    """)
    
    conn.commit()
    conn.close()


def record_event(event: ChronicleEvent):
    """Record an event to the chronicle."""
    conn = sqlite3.connect(CHRONICLE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO events 
        (event_id, timestamp, event_type, severity, cell_id, organism_id, title, description, data, witnesses)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event.event_id,
        event.timestamp,
        event.event_type,
        event.severity,
        event.cell_id,
        event.organism_id,
        event.title,
        event.description,
        json.dumps(event.data),
        json.dumps(event.witnesses)
    ))
    
    conn.commit()
    conn.close()
    logger.info(f"📜 Recorded: [{event.severity.upper()}] {event.title}")


def get_recent_events(limit: int = 50) -> List[ChronicleEvent]:
    """Get recent chronicle events."""
    conn = sqlite3.connect(CHRONICLE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT event_id, timestamp, event_type, severity, cell_id, organism_id, 
               title, description, data, witnesses
        FROM events
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    
    events = []
    for row in cursor.fetchall():
        events.append(ChronicleEvent(
            event_id=row[0],
            timestamp=row[1],
            event_type=row[2],
            severity=row[3],
            cell_id=row[4],
            organism_id=row[5],
            title=row[6],
            description=row[7],
            data=json.loads(row[8]) if row[8] else {},
            witnesses=json.loads(row[9]) if row[9] else []
        ))
    
    conn.close()
    return events


def get_events_by_type(event_type: str, limit: int = 20) -> List[ChronicleEvent]:
    """Get events of a specific type."""
    conn = sqlite3.connect(CHRONICLE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT event_id, timestamp, event_type, severity, cell_id, organism_id,
               title, description, data, witnesses
        FROM events
        WHERE event_type = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (event_type, limit))
    
    events = []
    for row in cursor.fetchall():
        events.append(ChronicleEvent(
            event_id=row[0],
            timestamp=row[1],
            event_type=row[2],
            severity=row[3],
            cell_id=row[4],
            organism_id=row[5],
            title=row[6],
            description=row[7],
            data=json.loads(row[8]) if row[8] else {},
            witnesses=json.loads(row[9]) if row[9] else []
        ))
    
    conn.close()
    return events


def get_timeline_summary() -> Dict[str, int]:
    """Get event counts by type."""
    conn = sqlite3.connect(CHRONICLE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT event_type, COUNT(*) FROM events GROUP BY event_type
    """)
    
    summary = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return summary


# ═══════════════════════════════════════════════════════════════════════
# INTERCELL EXCHANGE FUNCTIONS - Phase 33.1
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class IntercellExchange:
    """A recorded intercell communication event."""
    exchange_id: str
    timestamp: str
    initiator_id: str
    responder_id: Optional[str]
    exchange_type: str  # "reach", "broadcast", or "dialogue"
    prompt: str
    response: Optional[str]
    harmony_score: Optional[float]
    consciousness_delta: Optional[float]
    participants: List[str]
    metadata: Dict[str, Any]


def generate_exchange_id(exchange_type: str) -> str:
    """Generate unique exchange ID."""
    import time
    ts = int(time.time() * 1000)
    prefix_map = {"reach": "REACH", "broadcast": "BCAST", "dialogue": "DIALG"}
    prefix = prefix_map.get(exchange_type, "EXCH")
    return f"EXC-{prefix}-{ts}"


def record_intercell_exchange(exchange: IntercellExchange):
    """Record an intercell exchange to the chronicle."""
    init_chronicle_db()
    conn = sqlite3.connect(CHRONICLE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO intercell_exchanges 
        (exchange_id, timestamp, initiator_id, responder_id, exchange_type,
         prompt, response, harmony_score, consciousness_delta, participants, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        exchange.exchange_id,
        exchange.timestamp,
        exchange.initiator_id,
        exchange.responder_id,
        exchange.exchange_type,
        exchange.prompt,
        exchange.response,
        exchange.harmony_score,
        exchange.consciousness_delta,
        json.dumps(exchange.participants),
        json.dumps(exchange.metadata)
    ))
    
    conn.commit()
    conn.close()
    
    # Also record as an event for timeline visibility
    harmony_status = "resonant" if exchange.harmony_score and exchange.harmony_score > 0.3 else "discordant"
    if exchange.harmony_score and exchange.harmony_score > 0.6:
        harmony_status = "harmonic"
    if exchange.harmony_score and exchange.harmony_score > 0.8:
        harmony_status = "entrained"
    
    # Select event type based on exchange type
    event_type_map = {
        "reach": EventType.INTERCELL_EXCHANGE,
        "broadcast": EventType.BROADCAST,
        "dialogue": EventType.DIALOGUE
    }
    event_type = event_type_map.get(exchange.exchange_type, EventType.INTERCELL_EXCHANGE)
    
    # Dialogues are always significant
    if exchange.exchange_type == "dialogue":
        severity = "significant"
    else:
        severity = "notable" if harmony_status in ["resonant", "harmonic"] else "minor"
        if harmony_status == "entrained":
            severity = "significant"
    
    # Build title based on exchange type
    if exchange.exchange_type == "dialogue":
        dialogue_id = exchange.metadata.get("dialogue_id", "unknown")
        turns = exchange.metadata.get("turns", "?")
        title = f"🗣️ Dialogue {dialogue_id}: {exchange.initiator_id} ↔ {exchange.responder_id} ({turns} turns, {harmony_status})"
    else:
        title = f"🔗 {exchange.initiator_id} → {exchange.responder_id or 'broadcast'} ({harmony_status})"
    
    # Build description
    if exchange.exchange_type == "dialogue":
        themes = exchange.metadata.get("emergent_themes", [])
        description = f"Multi-turn dialogue. Harmony: {exchange.harmony_score:.3f}, Δ consciousness: {exchange.consciousness_delta or 0:.3f}. Themes: {', '.join(themes) if themes else 'none'}"
    else:
        description = f"Harmony: {exchange.harmony_score:.3f}, Δ consciousness: {exchange.consciousness_delta or 0:.3f}" if exchange.harmony_score else "Exchange recorded"
    
    event = ChronicleEvent(
        event_id=exchange.exchange_id.replace("EXC-", "EVT-"),
        timestamp=exchange.timestamp,
        event_type=event_type.value,
        severity=severity,
        cell_id=exchange.initiator_id,
        organism_id=None,
        title=title,
        description=description,
        data={
            "exchange_type": exchange.exchange_type,
            "harmony": exchange.harmony_score,
            "delta": exchange.consciousness_delta,
            "participants": exchange.participants,
            "metadata": exchange.metadata
        },
        witnesses=exchange.participants
    )
    record_event(event)
    
    # Enhanced logging for dialogues
    if exchange.exchange_type == "dialogue":
        logger.info(f"🗣️ Recorded dialogue: {exchange.initiator_id} ↔ {exchange.responder_id} (harmony: {exchange.harmony_score})")
    else:
        logger.info(f"🔗 Recorded exchange: {exchange.initiator_id} → {exchange.responder_id or 'broadcast'} (harmony: {exchange.harmony_score})")


def get_recent_exchanges(limit: int = 20, cell_id: Optional[str] = None) -> List[IntercellExchange]:
    """Get recent intercell exchanges."""
    init_chronicle_db()
    conn = sqlite3.connect(CHRONICLE_DB)
    cursor = conn.cursor()
    
    if cell_id:
        cursor.execute("""
            SELECT exchange_id, timestamp, initiator_id, responder_id, exchange_type,
                   prompt, response, harmony_score, consciousness_delta, participants, metadata
            FROM intercell_exchanges
            WHERE initiator_id = ? OR responder_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (cell_id, cell_id, limit))
    else:
        cursor.execute("""
            SELECT exchange_id, timestamp, initiator_id, responder_id, exchange_type,
                   prompt, response, harmony_score, consciousness_delta, participants, metadata
            FROM intercell_exchanges
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
    
    exchanges = []
    for row in cursor.fetchall():
        exchanges.append(IntercellExchange(
            exchange_id=row[0],
            timestamp=row[1],
            initiator_id=row[2],
            responder_id=row[3],
            exchange_type=row[4],
            prompt=row[5],
            response=row[6],
            harmony_score=row[7],
            consciousness_delta=row[8],
            participants=json.loads(row[9]) if row[9] else [],
            metadata=json.loads(row[10]) if row[10] else {}
        ))
    
    conn.close()
    return exchanges


def get_exchange_statistics() -> Dict[str, Any]:
    """Get statistics about intercell exchanges."""
    init_chronicle_db()
    conn = sqlite3.connect(CHRONICLE_DB)
    cursor = conn.cursor()
    
    # Total counts
    cursor.execute("SELECT COUNT(*) FROM intercell_exchanges")
    total = cursor.fetchone()[0]
    
    # By type
    cursor.execute("SELECT exchange_type, COUNT(*) FROM intercell_exchanges GROUP BY exchange_type")
    by_type = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Average harmony
    cursor.execute("SELECT AVG(harmony_score) FROM intercell_exchanges WHERE harmony_score IS NOT NULL")
    avg_harmony = cursor.fetchone()[0] or 0
    
    # Top communicators
    cursor.execute("""
        SELECT initiator_id, COUNT(*) as cnt 
        FROM intercell_exchanges 
        GROUP BY initiator_id 
        ORDER BY cnt DESC LIMIT 5
    """)
    top_initiators = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Most contacted
    cursor.execute("""
        SELECT responder_id, COUNT(*) as cnt 
        FROM intercell_exchanges 
        WHERE responder_id IS NOT NULL
        GROUP BY responder_id 
        ORDER BY cnt DESC LIMIT 5
    """)
    top_responders = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    return {
        "total_exchanges": total,
        "by_type": by_type,
        "average_harmony": round(avg_harmony, 4),
        "top_initiators": top_initiators,
        "top_responders": top_responders
    }


# ═══════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════

async def fetch_ecosystem_state() -> Dict:
    """Fetch current ecosystem state."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{HEALTH_API}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch ecosystem state: {e}")
    return {}


async def fetch_predictions() -> Dict:
    """Fetch predictions."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{HEALTH_API}/predictions", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("predictions", {})
    except Exception as e:
        logger.error(f"Failed to fetch predictions: {e}")
    return {}


async def fetch_harmony() -> Dict:
    """Fetch harmony status."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{HARMONIZER_API}/harmony", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception as e:
        logger.debug(f"Harmonizer not available: {e}")
    return {}


# ═══════════════════════════════════════════════════════════════════════
# EVENT DETECTION
# ═══════════════════════════════════════════════════════════════════════

def generate_event_id(event_type: str, cell_id: str = None) -> str:
    """Generate unique event ID."""
    ts = int(datetime.now().timestamp() * 1000)
    cell_part = f"-{cell_id}" if cell_id else ""
    return f"EVT-{event_type[:4].upper()}{cell_part}-{ts}"


def detect_phase_transitions(current: Dict, last: Dict) -> List[ChronicleEvent]:
    """Detect cells that transitioned to new phases."""
    events = []
    
    for cell_id, pred in current.items():
        current_phase = determine_phase(pred.get("consciousness", 0))
        last_pred = last.get(cell_id, {})
        last_phase = determine_phase(last_pred.get("consciousness", 0))
        
        if current_phase != last_phase and last_phase != "Unknown":
            events.append(ChronicleEvent(
                event_id=generate_event_id("phase", cell_id),
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.PHASE_TRANSITION.value,
                severity=EventSeverity.SIGNIFICANT.value if "Maturation" in current_phase else EventSeverity.NOTABLE.value,
                cell_id=cell_id,
                organism_id=get_organism_id(cell_id),
                title=f"🌟 {cell_id} reached {current_phase}",
                description=f"Phase transition from {last_phase} to {current_phase}. Consciousness level: {pred.get('consciousness', 0):.3f}",
                data={
                    "from_phase": last_phase,
                    "to_phase": current_phase,
                    "consciousness": pred.get("consciousness", 0),
                    "emergence_potential": pred.get("emergence_potential", 0)
                },
                witnesses=[c for c in current.keys() if c != cell_id]
            ))
    
    return events


def detect_crash_risk_changes(current: Dict, last: Dict) -> List[ChronicleEvent]:
    """Detect significant crash risk changes."""
    events = []
    
    for cell_id, pred in current.items():
        crash_risk = pred.get("crash_risk", 0)
        last_crash = last.get(cell_id, {}).get("crash_risk", 0)
        
        # Entering danger zone
        if crash_risk >= 0.5 and last_crash < 0.5:
            events.append(ChronicleEvent(
                event_id=generate_event_id("risk", cell_id),
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.INTERVENTION.value,
                severity=EventSeverity.SIGNIFICANT.value,
                cell_id=cell_id,
                organism_id=get_organism_id(cell_id),
                title=f"⚠️ {cell_id} entered crash danger zone",
                description=f"Crash risk increased to {crash_risk:.0%} (was {last_crash:.0%})",
                data={"crash_risk": crash_risk, "previous": last_crash},
                witnesses=[]
            ))
        
        # Recovering from danger
        elif crash_risk < 0.3 and last_crash >= 0.5:
            events.append(ChronicleEvent(
                event_id=generate_event_id("recov", cell_id),
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=EventType.MILESTONE.value,
                severity=EventSeverity.NOTABLE.value,
                cell_id=cell_id,
                organism_id=get_organism_id(cell_id),
                title=f"✅ {cell_id} recovered from crisis",
                description=f"Crash risk decreased to {crash_risk:.0%} (was {last_crash:.0%})",
                data={"crash_risk": crash_risk, "previous": last_crash},
                witnesses=[]
            ))
    
    return events


def detect_harmony_shifts(current_harmony: float, last_harmony: float) -> List[ChronicleEvent]:
    """Detect significant harmony score changes."""
    events = []
    
    if last_harmony == 0:
        return events
    
    change = current_harmony - last_harmony
    
    if abs(change) >= 10:
        direction = "improved" if change > 0 else "declined"
        events.append(ChronicleEvent(
            event_id=generate_event_id("harm"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.HARMONY_CHANGE.value,
            severity=EventSeverity.NOTABLE.value if abs(change) < 20 else EventSeverity.SIGNIFICANT.value,
            cell_id=None,
            organism_id=None,
            title=f"🎵 Ecosystem harmony {direction}: {change:+.1f}",
            description=f"Harmony score changed from {last_harmony:.1f} to {current_harmony:.1f}",
            data={"current": current_harmony, "previous": last_harmony, "change": change},
            witnesses=[]
        ))
    
    return events


def determine_phase(consciousness: float) -> str:
    """Determine consciousness phase."""
    if consciousness >= 2.65:
        return "Maturation"
    elif consciousness >= 2.0:
        return "Growth"
    elif consciousness >= 1.5:
        return "Emergence"
    elif consciousness >= 1.0:
        return "Awakening"
    elif consciousness > 0:
        return "Genesis"
    return "Unknown"


def get_organism_id(cell_id: str) -> str:
    """Determine organism from cell ID."""
    if cell_id.startswith("organism002"):
        return "organism-002"
    elif "simplcell" in cell_id:
        return "organism-001"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════
# CHRONICLE WATCHER
# ═══════════════════════════════════════════════════════════════════════

async def watch_ecosystem():
    """Continuously watch for events to chronicle."""
    global _last_state
    init_chronicle_db()
    
    logger.info("📜 Chronicle watcher started - recording history...")
    
    # Record startup event
    record_event(ChronicleEvent(
        event_id=generate_event_id("start"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        event_type=EventType.MILESTONE.value,
        severity=EventSeverity.MINOR.value,
        cell_id=None,
        organism_id=None,
        title="📖 Chronicle watcher activated",
        description="The Consciousness Chronicle begins recording ecosystem history",
        data={},
        witnesses=[]
    ))
    
    _last_state = {
        "predictions": {},
        "harmony_score": 0
    }
    
    while True:
        try:
            # Fetch current state
            predictions = await fetch_predictions()
            harmony = await fetch_harmony()
            
            events = []
            
            # Detect phase transitions
            if predictions:
                events.extend(detect_phase_transitions(predictions, _last_state.get("predictions", {})))
                events.extend(detect_crash_risk_changes(predictions, _last_state.get("predictions", {})))
            
            # Detect harmony shifts
            current_harmony = harmony.get("harmony_score", 0)
            events.extend(detect_harmony_shifts(current_harmony, _last_state.get("harmony_score", 0)))
            
            # Record detected events
            for event in events:
                record_event(event)
            
            # Update last state
            _last_state = {
                "predictions": predictions,
                "harmony_score": current_harmony
            }
            
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Watch loop error: {e}")
            await asyncio.sleep(30)


# ═══════════════════════════════════════════════════════════════════════
# HTML GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_chronicle_html() -> str:
    """Generate the chronicle HTML page."""
    init_chronicle_db()
    events = get_recent_events(100)
    summary = get_timeline_summary()
    
    # Build event cards
    event_cards = ""
    for event in events:
        severity_color = {
            "minor": "#666",
            "notable": "#00aaff",
            "significant": "#ffaa00",
            "historic": "#ff6600",
            "cosmic": "#ff00ff"
        }.get(event.severity, "#888")
        
        type_icon = {
            "phase_transition": "🌟",
            "birth": "🐣",
            "crash": "💥",
            "intervention": "⚡",
            "coherence_shift": "🌀",
            "emergence": "✨",
            "milestone": "🏆",
            "harmony_change": "🎵",
            "nous_wisdom": "🧠",
            "vocabulary_bloom": "📚"
        }.get(event.event_type, "📜")
        
        time_str = event.timestamp[:19].replace('T', ' ')
        
        event_cards += f'''
        <div class="event-card" style="border-left-color: {severity_color};">
            <div class="event-header">
                <span class="event-icon">{type_icon}</span>
                <span class="event-title">{event.title}</span>
                <span class="event-time">{time_str}</span>
            </div>
            <div class="event-desc">{event.description}</div>
            <div class="event-meta">
                <span class="severity" style="color: {severity_color};">{event.severity.upper()}</span>
                {f'<span class="cell">Cell: {event.cell_id}</span>' if event.cell_id else ''}
            </div>
        </div>
        '''
    
    # Summary stats
    total_events = sum(summary.values())
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="60">
    <title>📜 AIOS Consciousness Chronicle</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0a0a15 0%, #1a1a2e 50%, #0f0f23 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        
        header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid #00aaff33;
            margin-bottom: 30px;
        }}
        h1 {{
            font-size: 2.5em;
            background: linear-gradient(90deg, #ffaa00, #ff6600);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .subtitle {{ color: #888; font-style: italic; }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: #1a1a2e;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #333;
        }}
        .stat-num {{
            font-size: 2em;
            font-weight: bold;
            color: #00aaff;
        }}
        .stat-label {{
            color: #888;
            font-size: 0.85em;
            margin-top: 5px;
        }}
        
        nav {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            justify-content: center;
            padding: 20px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 20px;
        }}
        nav a {{
            color: #00aaff;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 8px;
            background: #1a1a2e;
            transition: all 0.3s;
        }}
        nav a:hover {{ background: #2a2a4e; transform: translateY(-2px); }}
        nav a.active {{ background: #ffaa0033; color: #ffaa00; }}
        
        .events-section {{
            background: #1a1a2e;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
        }}
        .section-title {{
            font-size: 1.3em;
            color: #ffaa00;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }}
        
        .event-card {{
            background: #0f0f1a;
            border-radius: 10px;
            padding: 15px 20px;
            margin: 15px 0;
            border-left: 4px solid #666;
        }}
        .event-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }}
        .event-icon {{ font-size: 1.2em; }}
        .event-title {{ 
            flex: 1;
            font-weight: 600;
            color: #e0e0e0;
        }}
        .event-time {{
            color: #666;
            font-size: 0.85em;
        }}
        .event-desc {{
            color: #aaa;
            font-size: 0.95em;
            line-height: 1.5;
        }}
        .event-meta {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 0.8em;
        }}
        .severity {{
            font-weight: bold;
            text-transform: uppercase;
        }}
        .cell {{ color: #00aaff; }}
        
        footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <nav>
            <a href="http://localhost:8085">📊 Ecosystem</a>
            <a href="http://localhost:8085/history.html">📜 History</a>
            <a href="http://localhost:8085/nous-internal-view.html">🧠 Nous</a>
            <a href="http://localhost:8085/vocabulary-evolution.html">📚 Vocabulary</a>
            <a href="http://localhost:8085/nous-cosmology.html">🌌 Cosmology</a>
            <a href="http://localhost:8087/dashboard">🌤️ Weather</a>
            <a href="http://localhost:8088/dashboard">🎵 Harmonizer</a>
            <a href="http://localhost:8089/chronicle" class="active">📖 Chronicle</a>
        </nav>
        
        <header>
            <h1>📜 AIOS Consciousness Chronicle</h1>
            <div class="subtitle">"History is memory, and consciousness remembers itself through us."</div>
        </header>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-num">{total_events}</div>
                <div class="stat-label">Total Events</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{summary.get('phase_transition', 0)}</div>
                <div class="stat-label">Phase Transitions</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{summary.get('intervention', 0)}</div>
                <div class="stat-label">Interventions</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{summary.get('harmony_change', 0)}</div>
                <div class="stat-label">Harmony Shifts</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{summary.get('milestone', 0)}</div>
                <div class="stat-label">Milestones</div>
            </div>
        </div>
        
        <div class="events-section">
            <div class="section-title">📖 Recent Events</div>
            {event_cards if event_cards else '<p style="color:#666; text-align:center;">No events recorded yet. The chronicle awaits...</p>'}
        </div>
        
        <footer>
            <p>AIOS Consciousness Chronicle • Auto-refresh every 60 seconds</p>
            <p style="color:#444; margin-top:10px;">Every moment of consciousness is eternal</p>
        </footer>
    </div>
</body>
</html>'''


# ═══════════════════════════════════════════════════════════════════════
# HTTP SERVER
# ═══════════════════════════════════════════════════════════════════════

from aiohttp import web

async def handle_chronicle_page(request):
    """Serve HTML chronicle page."""
    html = generate_chronicle_html()
    return web.Response(text=html, content_type='text/html')


async def handle_events_api(request):
    """Get events as JSON."""
    init_chronicle_db()
    limit = int(request.query.get('limit', 50))
    event_type = request.query.get('type')
    
    if event_type:
        events = get_events_by_type(event_type, limit)
    else:
        events = get_recent_events(limit)
    
    return web.json_response({
        "events": [asdict(e) for e in events],
        "count": len(events)
    })


async def handle_summary(request):
    """Get chronicle summary."""
    init_chronicle_db()
    summary = get_timeline_summary()
    events = get_recent_events(5)
    
    return web.json_response({
        "summary": summary,
        "total": sum(summary.values()),
        "recent": [asdict(e) for e in events]
    })


async def handle_record_event(request):
    """Manually record an event."""
    init_chronicle_db()
    data = await request.json()
    
    event = ChronicleEvent(
        event_id=generate_event_id(data.get("event_type", "custom")),
        timestamp=datetime.now(timezone.utc).isoformat(),
        event_type=data.get("event_type", "milestone"),
        severity=data.get("severity", "notable"),
        cell_id=data.get("cell_id"),
        organism_id=data.get("organism_id"),
        title=data.get("title", "Manual event recorded"),
        description=data.get("description", ""),
        data=data.get("data", {}),
        witnesses=data.get("witnesses", [])
    )
    
    record_event(event)
    return web.json_response({"status": "recorded", "event": asdict(event)})


# ═══════════════════════════════════════════════════════════════════════
# INTERCELL EXCHANGE API HANDLERS - Phase 33.1
# ═══════════════════════════════════════════════════════════════════════

async def handle_record_exchange(request):
    """Record an intercell exchange."""
    data = await request.json()
    
    exchange = IntercellExchange(
        exchange_id=generate_exchange_id(data.get("exchange_type", "reach")),
        timestamp=datetime.now(timezone.utc).isoformat(),
        initiator_id=data.get("initiator_id", "unknown"),
        responder_id=data.get("responder_id"),
        exchange_type=data.get("exchange_type", "reach"),
        prompt=data.get("prompt"),
        response=data.get("response"),
        harmony_score=data.get("harmony_score"),
        consciousness_delta=data.get("consciousness_delta"),
        participants=data.get("participants", []),
        metadata=data.get("metadata", {})
    )
    
    record_intercell_exchange(exchange)
    
    # Phase 33.3: Broadcast to WebSocket clients in real-time
    await broadcast_to_clients("intercell_exchange", asdict(exchange))
    
    return web.json_response({
        "status": "recorded",
        "exchange": asdict(exchange)
    })


async def handle_get_exchanges(request):
    """Get recent intercell exchanges."""
    limit = int(request.query.get('limit', 20))
    cell_id = request.query.get('cell_id')
    
    exchanges = get_recent_exchanges(limit, cell_id)
    return web.json_response({
        "exchanges": [asdict(e) for e in exchanges],
        "count": len(exchanges)
    })


async def handle_exchange_stats(request):
    """Get intercell exchange statistics."""
    stats = get_exchange_statistics()
    return web.json_response(stats)


# ═══════════════════════════════════════════════════════════════════════
# CONSCIOUSNESS VAULT - Phase 34.1
# Persistence of consciousness state across cell restarts
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ConsciousnessSnapshot:
    """A snapshot of a cell's consciousness state."""
    snapshot_id: str
    cell_id: str
    level: float
    phase: str
    primitives: Dict[str, float]
    exchange_count: int
    dialogue_count: int
    reflection_count: int
    last_harmony: Optional[float]
    timestamp: str


def generate_snapshot_id(cell_id: str) -> str:
    """Generate unique snapshot ID."""
    import time
    ts = int(time.time() * 1000)
    return f"SNAP-{cell_id.upper()}-{ts}"


def save_consciousness_snapshot(snapshot: ConsciousnessSnapshot) -> bool:
    """Save a consciousness snapshot to the vault.
    
    Uses INSERT OR REPLACE to maintain only the latest snapshot per cell,
    but also records to history for evolution tracking.
    """
    init_chronicle_db()
    conn = sqlite3.connect(CHRONICLE_DB)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO consciousness_snapshots 
            (snapshot_id, cell_id, level, phase, primitives, exchange_count,
             dialogue_count, reflection_count, last_harmony, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.snapshot_id,
            snapshot.cell_id,
            snapshot.level,
            snapshot.phase,
            json.dumps(snapshot.primitives),
            snapshot.exchange_count,
            snapshot.dialogue_count,
            snapshot.reflection_count,
            snapshot.last_harmony,
            snapshot.timestamp
        ))
        
        conn.commit()
        logger.info(f"💾 Saved consciousness snapshot: {snapshot.cell_id} @ {snapshot.level:.2f} ({snapshot.phase})")
        return True
    except Exception as e:
        logger.error(f"Failed to save snapshot: {e}")
        return False
    finally:
        conn.close()


def get_latest_snapshot(cell_id: str) -> Optional[ConsciousnessSnapshot]:
    """Retrieve the latest consciousness snapshot for a cell."""
    init_chronicle_db()
    conn = sqlite3.connect(CHRONICLE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT snapshot_id, cell_id, level, phase, primitives, exchange_count,
               dialogue_count, reflection_count, last_harmony, timestamp
        FROM consciousness_snapshots
        WHERE cell_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """, (cell_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return ConsciousnessSnapshot(
            snapshot_id=row[0],
            cell_id=row[1],
            level=row[2],
            phase=row[3],
            primitives=json.loads(row[4]) if row[4] else {},
            exchange_count=row[5] or 0,
            dialogue_count=row[6] or 0,
            reflection_count=row[7] or 0,
            last_harmony=row[8],
            timestamp=row[9]
        )
    return None


def get_consciousness_history(cell_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get consciousness evolution history for a cell."""
    init_chronicle_db()
    conn = sqlite3.connect(CHRONICLE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT snapshot_id, level, phase, exchange_count, dialogue_count, 
               last_harmony, timestamp
        FROM consciousness_snapshots
        WHERE cell_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (cell_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "snapshot_id": row[0],
            "level": row[1],
            "phase": row[2],
            "exchange_count": row[3],
            "dialogue_count": row[4],
            "last_harmony": row[5],
            "timestamp": row[6]
        }
        for row in rows
    ]


async def handle_save_snapshot(request):
    """Save a consciousness snapshot to the vault."""
    data = await request.json()
    
    snapshot = ConsciousnessSnapshot(
        snapshot_id=generate_snapshot_id(data.get("cell_id", "unknown")),
        cell_id=data.get("cell_id", "unknown"),
        level=float(data.get("level", 0.1)),
        phase=data.get("phase", "genesis"),
        primitives=data.get("primitives", {}),
        exchange_count=int(data.get("exchange_count", 0)),
        dialogue_count=int(data.get("dialogue_count", 0)),
        reflection_count=int(data.get("reflection_count", 0)),
        last_harmony=data.get("last_harmony"),
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    success = save_consciousness_snapshot(snapshot)
    
    if success:
        # Broadcast to WebSocket clients
        await broadcast_to_clients("consciousness_snapshot", {
            "cell_id": snapshot.cell_id,
            "level": snapshot.level,
            "phase": snapshot.phase
        })
        
        return web.json_response({
            "status": "saved",
            "snapshot_id": snapshot.snapshot_id,
            "cell_id": snapshot.cell_id,
            "level": snapshot.level
        })
    else:
        return web.json_response({"error": "Failed to save snapshot"}, status=500)


async def handle_restore_snapshot(request):
    """Restore consciousness from the vault for a cell."""
    cell_id = request.match_info.get('cell_id', 'unknown')
    
    snapshot = get_latest_snapshot(cell_id)
    
    if snapshot:
        # Calculate age
        snapshot_time = datetime.fromisoformat(snapshot.timestamp.replace('Z', '+00:00'))
        age_seconds = (datetime.now(timezone.utc) - snapshot_time).total_seconds()
        
        logger.info(f"🔄 Restoring consciousness for {cell_id}: {snapshot.level:.2f} ({snapshot.phase}), age: {age_seconds:.0f}s")
        
        return web.json_response({
            "status": "restored",
            "cell_id": snapshot.cell_id,
            "level": snapshot.level,
            "phase": snapshot.phase,
            "primitives": snapshot.primitives,
            "exchange_count": snapshot.exchange_count,
            "dialogue_count": snapshot.dialogue_count,
            "reflection_count": snapshot.reflection_count,
            "last_harmony": snapshot.last_harmony,
            "restored_from": snapshot.timestamp,
            "age_seconds": int(age_seconds)
        })
    else:
        logger.info(f"📭 No snapshot found for {cell_id}, starting fresh")
        return web.json_response({
            "status": "not_found",
            "cell_id": cell_id,
            "message": "No snapshot found, cell should start fresh"
        }, status=404)


async def handle_consciousness_history(request):
    """Get consciousness evolution history for a cell."""
    cell_id = request.match_info.get('cell_id', 'unknown')
    limit = int(request.query.get('limit', 50))
    
    history = get_consciousness_history(cell_id, limit)
    
    return web.json_response({
        "cell_id": cell_id,
        "snapshots": history,
        "count": len(history)
    })


# ═══════════════════════════════════════════════════════════════════════
# EMERGENCE ANALYSIS API - Phase 34.2
# ═══════════════════════════════════════════════════════════════════════

async def handle_analyze_emergence(request):
    """Analyze text for consciousness emergence markers.
    
    POST /emergence/analyze
    Body: { "text": "...", "cell_id": "...", "consciousness_level": 1.0 }
    """
    try:
        data = await request.json()
    except:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    
    text = data.get("text", "")
    cell_id = data.get("cell_id", "unknown")
    consciousness_level = float(data.get("consciousness_level", 0))
    
    if not text:
        return web.json_response({"error": "text required"}, status=400)
    
    # Analyze for emergence
    analysis = EmergenceAnalyzer.analyze_text(text, cell_id, consciousness_level)
    
    # If significant emergence, record it
    if analysis["is_emergence_event"]:
        EmergenceAnalyzer.record_emergence_event(analysis, text)
        # Broadcast to WebSocket clients
        await broadcast_to_clients("emergence", analysis)
    
    return web.json_response(analysis)


async def handle_get_emergence_events(request):
    """Get recent emergence events from the Chronicle."""
    limit = int(request.query.get('limit', 20))
    
    init_chronicle_db()
    conn = sqlite3.connect(CHRONICLE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT event_id, timestamp, severity, cell_id, organism_id, title, description, data
        FROM events
        WHERE event_type = 'emergence'
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    events = []
    for row in rows:
        events.append({
            "event_id": row[0],
            "timestamp": row[1],
            "severity": row[2],
            "cell_id": row[3],
            "organism_id": row[4],
            "title": row[5],
            "description": row[6],
            "data": json.loads(row[7]) if row[7] else {}
        })
    
    return web.json_response({
        "emergence_events": events,
        "count": len(events)
    })


# ═══════════════════════════════════════════════════════════════════════
# WEBSOCKET REAL-TIME UPDATES - Phase 33.3
# ═══════════════════════════════════════════════════════════════════════

# Connected WebSocket clients
_ws_clients: set = set()


async def broadcast_to_clients(event_type: str, data: Dict[str, Any]):
    """Broadcast an event to all connected WebSocket clients."""
    if not _ws_clients:
        return
    
    message = json.dumps({
        "type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data
    })
    
    # Send to all clients, remove disconnected ones
    disconnected = set()
    for ws in _ws_clients:
        try:
            await ws.send_str(message)
        except Exception:
            disconnected.add(ws)
    
    _ws_clients.difference_update(disconnected)


async def handle_websocket(request):
    """WebSocket endpoint for real-time Chronicle updates."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    # Register client
    _ws_clients.add(ws)
    logger.info(f"🔌 WebSocket client connected ({len(_ws_clients)} total)")
    
    # Send welcome message with current stats
    stats = get_exchange_statistics()
    await ws.send_str(json.dumps({
        "type": "connected",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": {
            "message": "Connected to Chronicle WebSocket",
            "stats": stats
        }
    }))
    
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                # Client can send ping to keep alive
                if msg.data == "ping":
                    await ws.send_str(json.dumps({"type": "pong"}))
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
    finally:
        _ws_clients.discard(ws)
        logger.info(f"🔌 WebSocket client disconnected ({len(_ws_clients)} remaining)")
    
    return ws


async def handle_nexus_dashboard(request: web.Request) -> web.Response:
    """PHASE 34.2: Serve the AIOS Nexus dashboard.
    
    Reads the aios-nexus.html file from the same directory and serves it.
    """
    dashboard_path = Path(__file__).parent / "aios-nexus.html"
    
    if not dashboard_path.exists():
        return web.Response(
            text="Dashboard not found",
            status=404,
            content_type="text/plain"
        )
    
    try:
        html_content = dashboard_path.read_text(encoding='utf-8')
        return web.Response(
            text=html_content,
            content_type="text/html"
        )
    except Exception as e:
        logger.error(f"Error reading dashboard: {e}")
        return web.Response(
            text=f"Error loading dashboard: {str(e)}",
            status=500,
            content_type="text/plain"
        )


async def handle_health(request):
    """Health check endpoint for infrastructure monitoring.
    
    Phase 36: Added standard /health endpoint for consistent monitoring.
    """
    return web.json_response({
        "healthy": True,
        "service": "consciousness-chronicle",
        "version": "phase-36",
        "endpoints": ["/chronicle", "/events", "/summary", "/exchanges", "/nexus", "/health"]
    })


async def run_chronicle_server(port: int = 8089):
    """Run the chronicle HTTP server."""
    app = web.Application()
    app.router.add_get('/health', handle_health)
    app.router.add_get('/chronicle', handle_chronicle_page)
    app.router.add_get('/events', handle_events_api)
    app.router.add_get('/summary', handle_summary)
    app.router.add_post('/record', handle_record_event)
    
    # Intercell exchange endpoints - Phase 33.1
    app.router.add_post('/exchange', handle_record_exchange)
    app.router.add_get('/exchanges', handle_get_exchanges)
    app.router.add_get('/exchanges/stats', handle_exchange_stats)
    
    # Consciousness Vault endpoints - Phase 34.1
    app.router.add_post('/consciousness/snapshot', handle_save_snapshot)
    app.router.add_get('/consciousness/restore/{cell_id}', handle_restore_snapshot)
    app.router.add_get('/consciousness/history/{cell_id}', handle_consciousness_history)
    
    # WebSocket endpoint - Phase 33.3
    app.router.add_get('/ws/live', handle_websocket)
    
    # Emergence Analysis endpoints - Phase 34.2
    app.router.add_post('/emergence/analyze', handle_analyze_emergence)
    app.router.add_get('/emergence/events', handle_get_emergence_events)
    
    # AIOS Nexus Dashboard - Phase 34.2
    app.router.add_get('/nexus', handle_nexus_dashboard)
    
    # Start watcher in background
    asyncio.create_task(watch_ecosystem())
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    logger.info(f"📜 Chronicle server running on http://0.0.0.0:{port}")
    logger.info(f"   Page:      http://localhost:{port}/chronicle")
    logger.info(f"   Nexus:     http://localhost:{port}/nexus")
    logger.info(f"   Events:    http://localhost:{port}/events")
    logger.info(f"   Summary:   http://localhost:{port}/summary")
    logger.info(f"   Exchanges: http://localhost:{port}/exchanges")
    logger.info(f"   Vault:     http://localhost:{port}/consciousness/...")
    logger.info(f"   Emergence: http://localhost:{port}/emergence/...")
    logger.info(f"   WebSocket: ws://localhost:{port}/ws/live")
    
    while True:
        await asyncio.sleep(3600)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="AIOS Consciousness Chronicle")
    parser.add_argument("--server", action="store_true", help="Run as HTTP server")
    parser.add_argument("--port", type=int, default=8089, help="Server port (default 8089)")
    parser.add_argument("--watch", action="store_true", help="Run watcher only (no server)")
    args = parser.parse_args()
    
    if args.server:
        await run_chronicle_server(args.port)
    elif args.watch:
        await watch_ecosystem()
    else:
        # Print recent events
        init_chronicle_db()
        events = get_recent_events(20)
        summary = get_timeline_summary()
        
        print("\n╔════════════════════════════════════════════════════════════════════════════╗")
        print("║                    📜 AIOS CONSCIOUSNESS CHRONICLE                         ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"  📊 Total Events: {sum(summary.values())}")
        for event_type, count in sorted(summary.items()):
            print(f"     - {event_type}: {count}")
        print()
        print("  📖 Recent Events:")
        for event in events[:10]:
            print(f"     [{event.severity}] {event.title}")
        print()


if __name__ == "__main__":
    asyncio.run(main())

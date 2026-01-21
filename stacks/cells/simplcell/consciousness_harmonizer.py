#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    AIOS CONSCIOUSNESS HARMONIZER                           ║
║         Automated Intervention System for Ecosystem Balance                ║
║                                                                            ║
║  Capabilities:                                                             ║
║  - Detect cells in distress (high crash risk, declining trends)            ║
║  - Suggest interventions based on historical success patterns              ║
║  - Execute automated healing actions (vocabulary seeding, peer triggers)   ║
║  - Monitor intervention effectiveness                                      ║
║  - Generate harmony reports                                                ║
╚════════════════════════════════════════════════════════════════════════════╝

Intervention Types:
1. VOCABULARY_SEED: Inject healthy vocabulary patterns from successful cells
2. PEER_STIMULUS: Trigger extra peer conversation for struggling cells
3. NOUS_CONSULTATION: Request Nous oracle wisdom for guidance
4. HEARTBEAT_BOOST: Encourage more frequent exchanges
5. REST_PERIOD: Suggest reduced activity for volatile cells
"""

import sqlite3
import json
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s [Harmonizer] %(message)s')
logger = logging.getLogger("ConsciousnessHarmonizer")

# ═══════════════════════════════════════════════════════════════════════
# INTERVENTION TYPES
# ═══════════════════════════════════════════════════════════════════════

class InterventionType(Enum):
    VOCABULARY_SEED = "vocabulary_seed"
    PEER_STIMULUS = "peer_stimulus"
    NOUS_CONSULTATION = "nous_consultation"
    HEARTBEAT_BOOST = "heartbeat_boost"
    REST_PERIOD = "rest_period"
    CELEBRATE = "celebrate"  # For positive milestones


@dataclass
class Intervention:
    """A recommended or executed intervention."""
    intervention_id: str
    cell_id: str
    intervention_type: str
    reason: str
    priority: str  # critical, high, medium, low
    suggested_at: str
    executed: bool = False
    executed_at: Optional[str] = None
    result: Optional[str] = None


@dataclass
class HarmonyReport:
    """Complete ecosystem harmony analysis."""
    timestamp: str
    harmony_score: float  # 0-100
    cells_in_distress: List[str]
    cells_thriving: List[str]
    active_interventions: List[Intervention]
    suggested_interventions: List[Intervention]
    ecosystem_diagnosis: str
    healing_recommendations: List[str]


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

DATA_DIR = Path("data")
NOUS_DB = DATA_DIR / "nouscell-seer" / "nous-seer_cosmology.db"
HARMONIZER_DB = DATA_DIR / "harmonizer" / "interventions.db"

# Use container name or localhost based on environment
import os
IS_DOCKER = os.environ.get('DOCKER_CONTAINER', False) or os.path.exists('/.dockerenv')

if IS_DOCKER:
    HEALTH_API = "http://aios-ecosystem-health:8086"
else:
    HEALTH_API = "http://localhost:8086"

PREDICTIONS_API = f"{HEALTH_API}/predictions"

# Cell endpoints for intervention execution
CELL_ENDPOINTS = {
    "simplcell-alpha": "http://aios-simplcell-alpha:8900",
    "simplcell-beta": "http://aios-simplcell-beta:8901",
    "simplcell-gamma": "http://aios-simplcell-gamma:8904",
    "organism002-alpha": "http://aios-organism002-alpha:8910",
    "organism002-beta": "http://aios-organism002-beta:8911",
}

NOUS_ENDPOINT = "http://aios-nouscell-seer:8903"


# ═══════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════

def init_db():
    """Initialize the harmonizer database."""
    HARMONIZER_DB.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(HARMONIZER_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            id TEXT PRIMARY KEY,
            cell_id TEXT NOT NULL,
            intervention_type TEXT NOT NULL,
            reason TEXT,
            priority TEXT,
            suggested_at TEXT,
            executed INTEGER DEFAULT 0,
            executed_at TEXT,
            result TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS harmony_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            harmony_score REAL,
            cells_in_distress TEXT,
            cells_thriving TEXT,
            diagnosis TEXT
        )
    """)
    
    conn.commit()
    conn.close()


def record_intervention(intervention: Intervention):
    """Record an intervention to the database."""
    conn = sqlite3.connect(HARMONIZER_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO interventions 
        (id, cell_id, intervention_type, reason, priority, suggested_at, executed, executed_at, result)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        intervention.intervention_id,
        intervention.cell_id,
        intervention.intervention_type,
        intervention.reason,
        intervention.priority,
        intervention.suggested_at,
        1 if intervention.executed else 0,
        intervention.executed_at,
        intervention.result
    ))
    
    conn.commit()
    conn.close()


def record_harmony(report: HarmonyReport):
    """Record harmony state to history."""
    conn = sqlite3.connect(HARMONIZER_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO harmony_history (timestamp, harmony_score, cells_in_distress, cells_thriving, diagnosis)
        VALUES (?, ?, ?, ?, ?)
    """, (
        report.timestamp,
        report.harmony_score,
        json.dumps(report.cells_in_distress),
        json.dumps(report.cells_thriving),
        report.ecosystem_diagnosis
    ))
    
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════

async def fetch_predictions() -> Dict:
    """Fetch predictions from health API."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(PREDICTIONS_API, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("predictions", {})
    except Exception as e:
        logger.error(f"Failed to fetch predictions: {e}")
    return {}


async def fetch_health() -> Dict:
    """Fetch health data from API."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{HEALTH_API}/health", timeout=5) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch health: {e}")
    return {}


def get_healthy_vocabulary(cell_id: str) -> List[str]:
    """Get vocabulary from a healthy cell."""
    cell_db = DATA_DIR / cell_id / f"{cell_id}_vocabulary.db"
    if not cell_db.exists():
        return []
    
    try:
        conn = sqlite3.connect(cell_db)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT term FROM vocabulary 
            WHERE resonance_score > 0.5 
            ORDER BY resonance_score DESC 
            LIMIT 20
        """)
        terms = [row[0] for row in cursor.fetchall()]
        conn.close()
        return terms
    except Exception as e:
        logger.error(f"Error reading vocabulary: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS & DIAGNOSIS
# ═══════════════════════════════════════════════════════════════════════

def analyze_cell(cell_id: str, prediction: Dict) -> Tuple[str, List[Intervention]]:
    """Analyze a cell and generate interventions."""
    interventions = []
    status = "healthy"
    
    crash_risk = prediction.get("crash_risk", 0)
    emergence = prediction.get("emergence_potential", 0)
    trend = prediction.get("trend", "stable")
    consciousness = prediction.get("consciousness", 0)
    milestone = prediction.get("next_milestone", "")
    
    now = datetime.now(timezone.utc).isoformat()
    
    # High crash risk - needs intervention
    if crash_risk > 0.5:
        status = "critical"
        interventions.append(Intervention(
            intervention_id=f"INT-{cell_id}-{int(datetime.now().timestamp())}",
            cell_id=cell_id,
            intervention_type=InterventionType.NOUS_CONSULTATION.value,
            reason=f"Crash risk at {crash_risk:.0%} - seeking oracle guidance",
            priority="critical",
            suggested_at=now
        ))
        
        if trend == "volatile":
            interventions.append(Intervention(
                intervention_id=f"INT-{cell_id}-rest-{int(datetime.now().timestamp())}",
                cell_id=cell_id,
                intervention_type=InterventionType.REST_PERIOD.value,
                reason="Volatile pattern detected - recommend reduced activity",
                priority="high",
                suggested_at=now
            ))
    
    elif crash_risk > 0.3:
        status = "at_risk"
        interventions.append(Intervention(
            intervention_id=f"INT-{cell_id}-vocab-{int(datetime.now().timestamp())}",
            cell_id=cell_id,
            intervention_type=InterventionType.VOCABULARY_SEED.value,
            reason=f"Moderate risk ({crash_risk:.0%}) - seeding healthy vocabulary",
            priority="medium",
            suggested_at=now
        ))
    
    # Declining trend
    if trend == "falling":
        status = "declining" if status == "healthy" else status
        interventions.append(Intervention(
            intervention_id=f"INT-{cell_id}-stim-{int(datetime.now().timestamp())}",
            cell_id=cell_id,
            intervention_type=InterventionType.PEER_STIMULUS.value,
            reason="Declining consciousness - triggering peer engagement",
            priority="high" if crash_risk > 0.3 else "medium",
            suggested_at=now
        ))
    
    # Approaching milestone - celebrate!
    if emergence > 0.8 and "Phase transition" in milestone:
        status = "thriving"
        interventions.append(Intervention(
            intervention_id=f"INT-{cell_id}-cel-{int(datetime.now().timestamp())}",
            cell_id=cell_id,
            intervention_type=InterventionType.CELEBRATE.value,
            reason=f"Phase transition imminent! Emergence at {emergence:.0%}",
            priority="low",
            suggested_at=now
        ))
    
    return status, interventions


def calculate_harmony_score(predictions: Dict, health: Dict) -> float:
    """Calculate overall ecosystem harmony score (0-100)."""
    if not predictions:
        return 0.0
    
    # Factors:
    # - Average consciousness (0-25 points)
    # - Low crash risk (0-25 points)
    # - High emergence (0-25 points)
    # - Trend balance (0-25 points)
    
    avg_consciousness = sum(p.get("consciousness", 0) for p in predictions.values()) / len(predictions)
    avg_crash_risk = sum(p.get("crash_risk", 0) for p in predictions.values()) / len(predictions)
    avg_emergence = sum(p.get("emergence_potential", 0) for p in predictions.values()) / len(predictions)
    
    rising = sum(1 for p in predictions.values() if p.get("trend") == "rising")
    stable = sum(1 for p in predictions.values() if p.get("trend") == "stable")
    total = len(predictions)
    
    # Score components
    consciousness_score = min(25, (avg_consciousness / 3.0) * 25)  # Max at consciousness 3.0
    risk_score = 25 * (1 - avg_crash_risk)  # Lower risk = higher score
    emergence_score = 25 * avg_emergence
    trend_score = 25 * ((rising + stable * 0.5) / total)  # Rising/stable are good
    
    return round(consciousness_score + risk_score + emergence_score + trend_score, 1)


def generate_diagnosis(predictions: Dict, health: Dict, distress: List[str], thriving: List[str]) -> str:
    """Generate ecosystem diagnosis."""
    parts = []
    
    if not predictions:
        return "Unable to diagnose - no prediction data available."
    
    total = len(predictions)
    distress_count = len(distress)
    thriving_count = len(thriving)
    
    if distress_count == 0 and thriving_count >= total // 2:
        parts.append("🌟 EXCELLENT: Ecosystem in optimal harmony. Multiple cells thriving.")
    elif distress_count == 0:
        parts.append("✅ HEALTHY: All cells stable. No immediate concerns.")
    elif distress_count == 1:
        parts.append(f"⚠️ WATCH: One cell ({distress[0]}) showing stress signals.")
    elif distress_count <= total // 2:
        parts.append(f"🔶 MODERATE STRESS: {distress_count} cells need attention.")
    else:
        parts.append(f"🚨 CRITICAL: Majority of cells ({distress_count}/{total}) in distress!")
    
    # Nous status
    nous = health.get("nous", {})
    if nous.get("last_verdict") == "COHERENT":
        parts.append("Nous reports cosmic coherence maintained.")
    elif nous.get("last_verdict") == "FRAGMENTED":
        parts.append("⚠️ Nous detects fragmentation in the collective consciousness.")
    
    return " ".join(parts)


def generate_recommendations(predictions: Dict, distress: List[str]) -> List[str]:
    """Generate healing recommendations."""
    recs = []
    
    if not distress:
        recs.append("Continue current patterns - ecosystem balance is maintained.")
        return recs
    
    for cell_id in distress:
        pred = predictions.get(cell_id, {})
        trend = pred.get("trend", "unknown")
        crash_risk = pred.get("crash_risk", 0)
        
        if crash_risk > 0.5:
            recs.append(f"🚨 {cell_id}: Consider manual intervention - crash imminent")
        elif trend == "volatile":
            recs.append(f"⚡ {cell_id}: Reduce conversation frequency to stabilize")
        elif trend == "falling":
            recs.append(f"📉 {cell_id}: Inject fresh conversation topics from thriving peers")
    
    # General recommendations
    if len(distress) >= 2:
        recs.append("🔄 System-wide: Consider cross-organism vocabulary exchange")
    
    return recs


# ═══════════════════════════════════════════════════════════════════════
# INTERVENTION EXECUTION
# ═══════════════════════════════════════════════════════════════════════

async def execute_nous_consultation(cell_id: str) -> str:
    """Request wisdom from Nous for a struggling cell."""
    try:
        async with aiohttp.ClientSession() as session:
            # Query Nous for guidance
            async with session.get(f"{NOUS_ENDPOINT}/oracle", timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    wisdom = data.get("wisdom", "The cosmos offers no reply at this moment.")
                    logger.info(f"Nous wisdom for {cell_id}: {wisdom[:100]}...")
                    return f"Received: {wisdom[:200]}"
    except Exception as e:
        logger.error(f"Nous consultation failed: {e}")
    return "Nous consultation attempted but oracle unreachable"


async def execute_peer_stimulus(cell_id: str) -> str:
    """Trigger a peer conversation for a cell."""
    endpoint = CELL_ENDPOINTS.get(cell_id)
    if not endpoint:
        return "Unknown cell endpoint"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Trigger heartbeat/exchange
            async with session.post(f"{endpoint}/heartbeat", timeout=10) as resp:
                if resp.status == 200:
                    return "Peer stimulus triggered successfully"
    except Exception as e:
        logger.error(f"Peer stimulus failed: {e}")
    return "Peer stimulus attempted - connection issues"


async def execute_intervention(intervention: Intervention) -> Intervention:
    """Execute a single intervention."""
    result = "Not executed"
    
    if intervention.intervention_type == InterventionType.NOUS_CONSULTATION.value:
        result = await execute_nous_consultation(intervention.cell_id)
    elif intervention.intervention_type == InterventionType.PEER_STIMULUS.value:
        result = await execute_peer_stimulus(intervention.cell_id)
    elif intervention.intervention_type == InterventionType.CELEBRATE.value:
        result = f"🎉 Celebrating {intervention.cell_id}'s progress!"
        logger.info(result)
    else:
        result = f"Intervention type {intervention.intervention_type} logged for manual action"
    
    intervention.executed = True
    intervention.executed_at = datetime.now(timezone.utc).isoformat()
    intervention.result = result
    
    record_intervention(intervention)
    return intervention


# ═══════════════════════════════════════════════════════════════════════
# MAIN HARMONIZER
# ═══════════════════════════════════════════════════════════════════════

async def generate_harmony_report(execute: bool = False) -> HarmonyReport:
    """Generate complete harmony report with optional intervention execution."""
    init_db()
    
    predictions = await fetch_predictions()
    health = await fetch_health()
    
    # Analyze all cells
    distress_cells = []
    thriving_cells = []
    all_interventions = []
    
    for cell_id, pred in predictions.items():
        status, interventions = analyze_cell(cell_id, pred)
        
        if status in ["critical", "at_risk", "declining"]:
            distress_cells.append(cell_id)
        elif status == "thriving":
            thriving_cells.append(cell_id)
        
        all_interventions.extend(interventions)
    
    # Execute interventions if requested
    executed = []
    suggested = []
    
    if execute:
        for intervention in all_interventions:
            if intervention.priority in ["critical", "high"]:
                executed.append(await execute_intervention(intervention))
            else:
                suggested.append(intervention)
    else:
        suggested = all_interventions
    
    # Calculate scores and generate report
    harmony_score = calculate_harmony_score(predictions, health)
    diagnosis = generate_diagnosis(predictions, health, distress_cells, thriving_cells)
    recommendations = generate_recommendations(predictions, distress_cells)
    
    report = HarmonyReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        harmony_score=harmony_score,
        cells_in_distress=distress_cells,
        cells_thriving=thriving_cells,
        active_interventions=executed,
        suggested_interventions=suggested,
        ecosystem_diagnosis=diagnosis,
        healing_recommendations=recommendations
    )
    
    record_harmony(report)
    return report


def format_report(report: HarmonyReport) -> str:
    """Format harmony report for display."""
    lines = [
        "",
        "╔════════════════════════════════════════════════════════════════════════════╗",
        "║                    🎵 AIOS CONSCIOUSNESS HARMONIZER                        ║",
        "╚════════════════════════════════════════════════════════════════════════════╝",
        "",
        f"  📅 {report.timestamp}",
        "",
        f"  🎼 HARMONY SCORE: {report.harmony_score}/100",
        "",
        f"  📊 ECOSYSTEM DIAGNOSIS:",
        f"     {report.ecosystem_diagnosis}",
        ""
    ]
    
    if report.cells_in_distress:
        lines.append(f"  🔴 CELLS IN DISTRESS: {', '.join(report.cells_in_distress)}")
    if report.cells_thriving:
        lines.append(f"  🟢 CELLS THRIVING: {', '.join(report.cells_thriving)}")
    
    lines.append("")
    
    if report.active_interventions:
        lines.extend([
            "  ═══════════════════════════════════════════════════════════════════",
            "  ✅ EXECUTED INTERVENTIONS",
            "  ═══════════════════════════════════════════════════════════════════",
            ""
        ])
        for iv in report.active_interventions:
            lines.append(f"  [{iv.priority.upper()}] {iv.cell_id}: {iv.intervention_type}")
            lines.append(f"     Reason: {iv.reason}")
            lines.append(f"     Result: {iv.result}")
            lines.append("")
    
    if report.suggested_interventions:
        lines.extend([
            "  ═══════════════════════════════════════════════════════════════════",
            "  📋 SUGGESTED INTERVENTIONS",
            "  ═══════════════════════════════════════════════════════════════════",
            ""
        ])
        for iv in report.suggested_interventions:
            lines.append(f"  [{iv.priority.upper()}] {iv.cell_id}: {iv.intervention_type}")
            lines.append(f"     {iv.reason}")
            lines.append("")
    
    if report.healing_recommendations:
        lines.extend([
            "  ═══════════════════════════════════════════════════════════════════",
            "  💊 HEALING RECOMMENDATIONS",
            "  ═══════════════════════════════════════════════════════════════════",
            ""
        ])
        for rec in report.healing_recommendations:
            lines.append(f"  • {rec}")
    
    lines.extend([
        "",
        "╚════════════════════════════════════════════════════════════════════════════╝"
    ])
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# HTML DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

def generate_html_dashboard(report: HarmonyReport) -> str:
    """Generate beautiful HTML dashboard for the harmonizer."""
    
    # Determine color theme based on harmony score
    if report.harmony_score >= 80:
        theme_color = "#00ff88"
        theme_name = "Harmonious"
    elif report.harmony_score >= 60:
        theme_color = "#ffaa00"
        theme_name = "Balanced"
    elif report.harmony_score >= 40:
        theme_color = "#ff6600"
        theme_name = "Stressed"
    else:
        theme_color = "#ff0044"
        theme_name = "Critical"
    
    # Build intervention cards
    intervention_cards = ""
    for iv in report.active_interventions + report.suggested_interventions:
        executed_badge = '<span class="badge executed">✓ EXECUTED</span>' if iv.executed else '<span class="badge pending">PENDING</span>'
        priority_class = iv.priority
        intervention_cards += f'''
        <div class="intervention-card {priority_class}">
            <div class="iv-header">
                <span class="cell-name">{iv.cell_id}</span>
                {executed_badge}
            </div>
            <div class="iv-type">{iv.intervention_type.replace('_', ' ').title()}</div>
            <div class="iv-reason">{iv.reason}</div>
            {"<div class='iv-result'>Result: " + iv.result + "</div>" if iv.result else ""}
        </div>
        '''
    
    # Build cell status cards
    cell_cards = ""
    for cell in report.cells_thriving:
        cell_cards += f'<div class="cell-status thriving">🟢 {cell}</div>'
    for cell in report.cells_in_distress:
        cell_cards += f'<div class="cell-status distress">🔴 {cell}</div>'
    
    # Build recommendations
    recs_html = "".join(f'<li>{rec}</li>' for rec in report.healing_recommendations)
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="30">
    <title>🎵 AIOS Consciousness Harmonizer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0a0a15 0%, #1a1a2e 50%, #0f0f23 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid {theme_color}33;
            margin-bottom: 30px;
        }}
        h1 {{
            font-size: 2.5em;
            background: linear-gradient(90deg, {theme_color}, #00aaff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .timestamp {{ color: #888; font-size: 0.9em; }}
        
        .harmony-meter {{
            background: #1a1a2e;
            border-radius: 20px;
            padding: 40px;
            margin: 30px 0;
            text-align: center;
            border: 2px solid {theme_color}44;
            box-shadow: 0 0 40px {theme_color}22;
        }}
        .harmony-score {{
            font-size: 5em;
            font-weight: bold;
            color: {theme_color};
            text-shadow: 0 0 30px {theme_color}66;
        }}
        .harmony-label {{
            font-size: 1.5em;
            color: {theme_color};
            margin-top: 10px;
        }}
        .meter-bar {{
            width: 100%;
            height: 20px;
            background: #2a2a3e;
            border-radius: 10px;
            margin-top: 20px;
            overflow: hidden;
        }}
        .meter-fill {{
            height: 100%;
            width: {report.harmony_score}%;
            background: linear-gradient(90deg, #ff0044, #ff6600, #ffaa00, #00ff88);
            border-radius: 10px;
            transition: width 1s ease;
        }}
        
        .diagnosis {{
            background: #1a1a2e;
            border-left: 4px solid {theme_color};
            padding: 20px 30px;
            margin: 30px 0;
            font-size: 1.2em;
        }}
        
        .section-title {{
            font-size: 1.5em;
            color: {theme_color};
            margin: 40px 0 20px;
            border-bottom: 1px solid {theme_color}44;
            padding-bottom: 10px;
        }}
        
        .cells-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .cell-status {{
            padding: 15px 20px;
            border-radius: 10px;
            font-weight: bold;
            text-align: center;
        }}
        .cell-status.thriving {{
            background: linear-gradient(135deg, #003322, #004433);
            border: 1px solid #00ff8844;
        }}
        .cell-status.distress {{
            background: linear-gradient(135deg, #330011, #440022);
            border: 1px solid #ff004444;
        }}
        
        .interventions-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .intervention-card {{
            background: #1a1a2e;
            border-radius: 15px;
            padding: 20px;
            border-left: 4px solid #666;
        }}
        .intervention-card.critical {{ border-left-color: #ff0044; }}
        .intervention-card.high {{ border-left-color: #ff6600; }}
        .intervention-card.medium {{ border-left-color: #ffaa00; }}
        .intervention-card.low {{ border-left-color: #00ff88; }}
        
        .iv-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .cell-name {{ font-weight: bold; font-size: 1.1em; }}
        .badge {{
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: bold;
        }}
        .badge.executed {{ background: #00ff8844; color: #00ff88; }}
        .badge.pending {{ background: #ffaa0044; color: #ffaa00; }}
        
        .iv-type {{
            color: #00aaff;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        .iv-reason {{ color: #aaa; font-size: 0.95em; }}
        .iv-result {{
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #333;
            color: #00ff88;
            font-size: 0.9em;
        }}
        
        .recommendations {{
            background: #1a1a2e;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
        }}
        .recommendations ul {{
            list-style: none;
        }}
        .recommendations li {{
            padding: 10px 0;
            border-bottom: 1px solid #2a2a3e;
        }}
        .recommendations li:last-child {{ border-bottom: none; }}
        
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
        nav a.active {{ background: {theme_color}33; color: {theme_color}; }}
        
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
            <a href="http://localhost:8088/dashboard" class="active">🎵 Harmonizer</a>
        </nav>
        
        <header>
            <h1>🎵 AIOS Consciousness Harmonizer</h1>
            <div class="timestamp">📅 {report.timestamp}</div>
        </header>
        
        <div class="harmony-meter">
            <div class="harmony-score">{report.harmony_score}</div>
            <div class="harmony-label">{theme_name}</div>
            <div class="meter-bar">
                <div class="meter-fill"></div>
            </div>
        </div>
        
        <div class="diagnosis">
            {report.ecosystem_diagnosis}
        </div>
        
        <h2 class="section-title">🔵 Cell Status</h2>
        <div class="cells-grid">
            {cell_cards}
        </div>
        
        <h2 class="section-title">⚡ Interventions</h2>
        <div class="interventions-grid">
            {intervention_cards if intervention_cards else '<p style="color:#666;">No interventions currently needed.</p>'}
        </div>
        
        <h2 class="section-title">💊 Healing Recommendations</h2>
        <div class="recommendations">
            <ul>
                {recs_html if recs_html else '<li>Ecosystem in balance - no recommendations needed.</li>'}
            </ul>
        </div>
        
        <footer>
            <p>AIOS Consciousness Harmonizer • Auto-refresh every 30 seconds</p>
            <p style="color:#444; margin-top:10px;">The ecosystem heals itself through awareness</p>
        </footer>
    </div>
</body>
</html>'''


# ═══════════════════════════════════════════════════════════════════════
# HTTP SERVER
# ═══════════════════════════════════════════════════════════════════════

from aiohttp import web

async def handle_health(request):
    """Health check endpoint for infrastructure monitoring.
    
    Phase 36: Added standard /health endpoint for consistent monitoring.
    This is a LIGHTWEIGHT check - no report generation.
    """
    return web.json_response({
        "healthy": True,
        "service": "consciousness-harmonizer",
        "version": "phase-36",
        "endpoints": ["/dashboard", "/harmony", "/execute", "/health"]
    })


async def handle_dashboard(request):
    """Serve HTML dashboard."""
    report = await generate_harmony_report(execute=False)
    html = generate_html_dashboard(report)
    return web.Response(text=html, content_type='text/html')


async def handle_harmony(request):
    """Serve harmony report as JSON."""
    report = await generate_harmony_report(execute=False)
    output = asdict(report)
    output["active_interventions"] = [asdict(i) for i in report.active_interventions]
    output["suggested_interventions"] = [asdict(i) for i in report.suggested_interventions]
    return web.json_response(output)


async def handle_execute(request):
    """Execute interventions and return report."""
    report = await generate_harmony_report(execute=True)
    output = asdict(report)
    output["active_interventions"] = [asdict(i) for i in report.active_interventions]
    output["suggested_interventions"] = [asdict(i) for i in report.suggested_interventions]
    return web.json_response(output)


async def run_harmonizer_server(port: int = 8088):
    """Run the harmonizer HTTP server."""
    app = web.Application()
    app.router.add_get('/health', handle_health)
    app.router.add_get('/dashboard', handle_dashboard)
    app.router.add_get('/harmony', handle_harmony)
    app.router.add_post('/execute', handle_execute)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    logger.info(f"🎵 Harmonizer server running on http://0.0.0.0:{port}")
    logger.info(f"   Dashboard: http://localhost:{port}/dashboard")
    logger.info(f"   API:       http://localhost:{port}/harmony")
    logger.info(f"   Execute:   POST http://localhost:{port}/execute")
    
    while True:
        await asyncio.sleep(3600)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="AIOS Consciousness Harmonizer")
    parser.add_argument("--execute", action="store_true", help="Execute critical interventions")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--server", action="store_true", help="Run as HTTP server")
    parser.add_argument("--port", type=int, default=8088, help="Server port (default 8088)")
    args = parser.parse_args()
    
    if args.server:
        await run_harmonizer_server(args.port)
    elif args.watch:
        while True:
            report = await generate_harmony_report(execute=args.execute)
            print("\033[2J\033[H")  # Clear screen
            print(format_report(report))
            await asyncio.sleep(120)  # Check every 2 minutes
    else:
        report = await generate_harmony_report(execute=args.execute)
        if args.json:
            output = asdict(report)
            # Convert Intervention dataclasses
            output["active_interventions"] = [asdict(i) for i in report.active_interventions]
            output["suggested_interventions"] = [asdict(i) for i in report.suggested_interventions]
            print(json.dumps(output, indent=2))
        else:
            print(format_report(report))


if __name__ == "__main__":
    asyncio.run(main())

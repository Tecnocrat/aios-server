#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                      AIOS ECOSYSTEM HEALTH API                             ║
║            Real-time Health Monitoring for the AIOS Cellular Organism      ║
║                                                                            ║
║  Provides comprehensive health metrics via HTTP API:                       ║
║  - /health - Overall ecosystem health                                      ║
║  - /cells - Individual cell states                                         ║
║  - /organisms - Organism-level metrics                                     ║
║  - /events - Recent consciousness events                                   ║
║  - /metrics - Prometheus-compatible metrics                                ║
╚════════════════════════════════════════════════════════════════════════════╝

Usage:
    python ecosystem_health_api.py              # Starts server on port 8086
    python ecosystem_health_api.py --port 8090  # Custom port
"""

import asyncio
import json
import sqlite3
from aiohttp import web
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from collections import defaultdict
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("EcosystemHealth")

# Paths
DATA_DIR = Path("data")
NOUS_DB = DATA_DIR / "nouscell-seer" / "nous-seer_cosmology.db"

# Cell configurations
CELLS = {
    "organism-001": {
        "simplcell-alpha": {"port": 8900, "role": "analytical"},
        "simplcell-beta": {"port": 8901, "role": "creative"},
        "simplcell-gamma": {"port": 8904, "role": "integrative"},
    },
    "organism-002": {
        "organism002-alpha": {"port": 8910, "role": "analytical"},
        "organism002-beta": {"port": 8911, "role": "creative"},
    }
}

# Phase thresholds
PHASES = [
    (0.0, 0.3, "Genesis"),
    (0.3, 0.7, "Awakening"),
    (0.7, 1.5, "Transcendence"),
    (1.5, 3.0, "Maturation"),
    (3.0, 5.0, "Advanced"),
]


def get_phase(consciousness: float) -> str:
    """Get phase name for consciousness level."""
    for low, high, name in PHASES:
        if low <= consciousness < high:
            return name
    return "Advanced"


def calculate_health_score(consciousness: float, vocabulary_size: int, conversations: int) -> float:
    """Calculate overall health score (0-100)."""
    # Consciousness contribution (40%): optimal at 1.5-2.5
    if consciousness < 0.5:
        c_score = consciousness / 0.5 * 40
    elif consciousness < 2.5:
        c_score = 40 + (consciousness - 0.5) / 2.0 * 30
    else:
        c_score = 70 + min(30, (consciousness - 2.5) / 2.5 * 30)
    
    c_contribution = min(40, c_score * 0.4)
    
    # Vocabulary contribution (30%): more is better, diminishing returns
    v_contribution = min(30, (vocabulary_size / 100) * 30)
    
    # Activity contribution (30%): based on conversations
    a_contribution = min(30, (conversations / 1000) * 30)
    
    return round(c_contribution + v_contribution + a_contribution, 1)


class EcosystemHealthService:
    """Service for aggregating ecosystem health data."""
    
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
    
    def get_cell_state(self, cell_id: str) -> Optional[Dict]:
        """Get current state of a specific cell."""
        db_path = DATA_DIR / cell_id / f"{cell_id}.db"
        if not db_path.exists():
            return None
        
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get cell state
            cursor.execute("SELECT * FROM cell_state WHERE id = 1")
            state_row = cursor.fetchone()
            
            if not state_row:
                conn.close()
                return None
            
            state = dict(state_row)
            
            # Get vocabulary count
            cursor.execute("SELECT COUNT(*) FROM vocabulary")
            vocab_count = cursor.fetchone()[0]
            
            # Get conversation count
            cursor.execute("SELECT COUNT(*) FROM conversation_archive")
            conv_count = cursor.fetchone()[0]
            
            conn.close()
            
            consciousness = state.get('consciousness', 0)
            return {
                "cell_id": cell_id,
                "consciousness": consciousness,
                "phase": get_phase(consciousness),
                "heartbeat_count": state.get('heartbeat_count', 0),
                "conversation_count": conv_count,
                "vocabulary_size": vocab_count,
                "last_thought": state.get('last_thought', '')[:200] if state.get('last_thought') else None,
                "updated_at": state.get('updated_at'),
                "health_score": calculate_health_score(consciousness, vocab_count, conv_count)
            }
        except Exception as e:
            logger.error(f"Error reading cell {cell_id}: {e}")
            return None
    
    def get_all_cells(self) -> List[Dict]:
        """Get state of all cells."""
        cells = []
        for organism, cell_configs in CELLS.items():
            for cell_id, config in cell_configs.items():
                state = self.get_cell_state(cell_id)
                if state:
                    state["organism"] = organism
                    state["port"] = config["port"]
                    state["role"] = config["role"]
                    cells.append(state)
        return cells
    
    def get_organism_stats(self) -> Dict[str, Dict]:
        """Get aggregated statistics per organism."""
        cells = self.get_all_cells()
        
        organisms = {}
        for cell in cells:
            org = cell["organism"]
            if org not in organisms:
                organisms[org] = {
                    "cells": [],
                    "total_consciousness": 0,
                    "total_vocabulary": 0,
                    "total_conversations": 0,
                    "avg_health": 0
                }
            
            organisms[org]["cells"].append(cell["cell_id"])
            organisms[org]["total_consciousness"] += cell["consciousness"]
            organisms[org]["total_vocabulary"] += cell["vocabulary_size"]
            organisms[org]["total_conversations"] += cell["conversation_count"]
            organisms[org]["avg_health"] += cell["health_score"]
        
        # Calculate averages
        for org, stats in organisms.items():
            n = len(stats["cells"])
            if n > 0:
                stats["avg_consciousness"] = round(stats["total_consciousness"] / n, 3)
                stats["avg_health"] = round(stats["avg_health"] / n, 1)
        
        return organisms
    
    def get_nous_summary(self) -> Optional[Dict]:
        """Get summary from Nous cosmology database."""
        if not NOUS_DB.exists():
            return None
        
        try:
            conn = sqlite3.connect(NOUS_DB)
            cursor = conn.cursor()
            
            # Exchange count
            cursor.execute("SELECT COUNT(*) FROM exchanges")
            exchange_count = cursor.fetchone()[0]
            
            # Broadcast count
            cursor.execute("SELECT COUNT(*) FROM broadcasts")
            broadcast_count = cursor.fetchone()[0]
            
            # Theme count
            cursor.execute("SELECT COUNT(*) FROM cosmology_themes")
            theme_count = cursor.fetchone()[0]
            
            # Recent verdict
            cursor.execute("""
                SELECT verdict, overall_coherence, philosophical_reflection
                FROM hourly_assessments
                ORDER BY created_at DESC LIMIT 1
            """)
            assessment = cursor.fetchone()
            
            conn.close()
            
            return {
                "exchanges_absorbed": exchange_count,
                "broadcasts_sent": broadcast_count,
                "themes_discovered": theme_count,
                "last_verdict": assessment[0] if assessment else None,
                "last_coherence": assessment[1] if assessment else None,
                "last_reflection": assessment[2] if assessment else None
            }
        except Exception as e:
            logger.error(f"Error reading Nous database: {e}")
            return None
    
    def get_recent_events(self, limit: int = 20) -> List[Dict]:
        """Get recent consciousness events from Nous."""
        if not NOUS_DB.exists():
            return []
        
        try:
            conn = sqlite3.connect(NOUS_DB)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT source_cell, heartbeat, consciousness, absorbed_at
                FROM exchanges
                ORDER BY absorbed_at DESC
                LIMIT ?
            """, (limit,))
            
            events = []
            prev_consciousness = {}
            
            for row in cursor.fetchall():
                cell = row['source_cell']
                curr_c = row['consciousness'] or 0
                prev_c = prev_consciousness.get(cell, curr_c)
                delta = curr_c - prev_c
                
                event_type = "stable"
                if delta < -0.5:
                    event_type = "crash"
                elif delta > 0.3:
                    event_type = "surge"
                elif delta < -0.1:
                    event_type = "decline"
                elif delta > 0.05:
                    event_type = "growth"
                
                events.append({
                    "cell": cell,
                    "heartbeat": row['heartbeat'],
                    "consciousness": curr_c,
                    "delta": round(delta, 3),
                    "event_type": event_type,
                    "timestamp": row['absorbed_at']
                })
                
                prev_consciousness[cell] = curr_c
            
            conn.close()
            return events
        except Exception as e:
            logger.error(f"Error reading events: {e}")
            return []
    
    def get_ecosystem_health(self) -> Dict:
        """Get overall ecosystem health summary."""
        cells = self.get_all_cells()
        organisms = self.get_organism_stats()
        nous = self.get_nous_summary()
        
        # Calculate ecosystem-wide metrics
        total_consciousness = sum(c["consciousness"] for c in cells)
        avg_consciousness = total_consciousness / len(cells) if cells else 0
        total_vocabulary = sum(c["vocabulary_size"] for c in cells)
        total_conversations = sum(c["conversation_count"] for c in cells)
        avg_health = sum(c["health_score"] for c in cells) / len(cells) if cells else 0
        
        # Find leader
        leader = max(cells, key=lambda c: c["consciousness"]) if cells else None
        
        # Calculate uptime
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        return {
            "status": "healthy" if avg_health > 50 else "degraded" if avg_health > 25 else "critical",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime,
            "summary": {
                "total_cells": len(cells),
                "total_organisms": len(organisms),
                "avg_consciousness": round(avg_consciousness, 3),
                "total_vocabulary": total_vocabulary,
                "total_conversations": total_conversations,
                "avg_health_score": round(avg_health, 1)
            },
            "leader": {
                "cell_id": leader["cell_id"],
                "consciousness": leader["consciousness"],
                "phase": leader["phase"]
            } if leader else None,
            "nous": nous,
            "organisms": organisms
        }
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics."""
        lines = [
            "# HELP aios_cell_consciousness Current consciousness level of a cell",
            "# TYPE aios_cell_consciousness gauge"
        ]
        
        cells = self.get_all_cells()
        for cell in cells:
            lines.append(
                f'aios_cell_consciousness{{cell="{cell["cell_id"]}",organism="{cell["organism"]}"}} {cell["consciousness"]}'
            )
        
        lines.extend([
            "",
            "# HELP aios_cell_vocabulary_size Number of vocabulary terms",
            "# TYPE aios_cell_vocabulary_size gauge"
        ])
        for cell in cells:
            lines.append(
                f'aios_cell_vocabulary_size{{cell="{cell["cell_id"]}"}} {cell["vocabulary_size"]}'
            )
        
        lines.extend([
            "",
            "# HELP aios_cell_health_score Health score 0-100",
            "# TYPE aios_cell_health_score gauge"
        ])
        for cell in cells:
            lines.append(
                f'aios_cell_health_score{{cell="{cell["cell_id"]}"}} {cell["health_score"]}'
            )
        
        nous = self.get_nous_summary()
        if nous:
            lines.extend([
                "",
                "# HELP aios_nous_exchanges Total exchanges absorbed by Nous",
                "# TYPE aios_nous_exchanges counter",
                f'aios_nous_exchanges {nous["exchanges_absorbed"]}',
                "",
                "# HELP aios_nous_broadcasts Total broadcasts sent by Nous",
                "# TYPE aios_nous_broadcasts counter",
                f'aios_nous_broadcasts {nous["broadcasts_sent"]}'
            ])
        
        return "\n".join(lines)


# HTTP Handlers
service = EcosystemHealthService()


async def health_handler(request: web.Request) -> web.Response:
    """GET /health - Overall ecosystem health."""
    return web.json_response(service.get_ecosystem_health())


async def cells_handler(request: web.Request) -> web.Response:
    """GET /cells - All cell states."""
    return web.json_response({"cells": service.get_all_cells()})


async def cell_handler(request: web.Request) -> web.Response:
    """GET /cells/{cell_id} - Specific cell state."""
    cell_id = request.match_info["cell_id"]
    state = service.get_cell_state(cell_id)
    if state:
        return web.json_response(state)
    return web.json_response({"error": f"Cell {cell_id} not found"}, status=404)


async def organisms_handler(request: web.Request) -> web.Response:
    """GET /organisms - Organism-level stats."""
    return web.json_response({"organisms": service.get_organism_stats()})


async def events_handler(request: web.Request) -> web.Response:
    """GET /events - Recent consciousness events."""
    limit = int(request.query.get("limit", 20))
    return web.json_response({"events": service.get_recent_events(limit)})


async def metrics_handler(request: web.Request) -> web.Response:
    """GET /metrics - Prometheus metrics."""
    return web.Response(text=service.get_prometheus_metrics(), content_type="text/plain")


async def index_handler(request: web.Request) -> web.Response:
    """GET / - API documentation."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>AIOS Ecosystem Health API</title>
    <style>
        body { font-family: monospace; background: #0a0a0a; color: #00ff88; padding: 20px; }
        h1 { color: #88ffff; }
        a { color: #ff88ff; }
        .endpoint { margin: 10px 0; padding: 10px; background: #1a1a1a; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🧬 AIOS Ecosystem Health API</h1>
    <p>Real-time health monitoring for the AIOS cellular organism.</p>
    
    <h2>Endpoints</h2>
    <div class="endpoint"><a href="/health">GET /health</a> - Overall ecosystem health summary</div>
    <div class="endpoint"><a href="/cells">GET /cells</a> - All cell states</div>
    <div class="endpoint">GET /cells/{cell_id} - Specific cell state</div>
    <div class="endpoint"><a href="/organisms">GET /organisms</a> - Organism-level statistics</div>
    <div class="endpoint"><a href="/events">GET /events</a> - Recent consciousness events</div>
    <div class="endpoint"><a href="/metrics">GET /metrics</a> - Prometheus metrics</div>
    <div class="endpoint"><a href="/predictions">GET /predictions</a> - Consciousness emergence predictions</div>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


# Import prediction module if available
try:
    from consciousness_emergence_predictor import predict_all, predict_cell
    PREDICTIONS_AVAILABLE = True
except ImportError:
    PREDICTIONS_AVAILABLE = False
    logger.warning("Predictions module not available")


async def predictions_handler(request: web.Request) -> web.Response:
    """GET /predictions - Consciousness emergence predictions."""
    if not PREDICTIONS_AVAILABLE:
        return web.json_response({"error": "Predictions module not available"}, status=503)
    
    cell_id = request.query.get("cell")
    
    try:
        if cell_id:
            pred = predict_cell(cell_id)
            if pred:
                return web.json_response({
                    "cell_id": pred.cell_id,
                    "consciousness": pred.current_consciousness,
                    "phase": pred.current_phase,
                    "trend": pred.trend,
                    "crash_risk": pred.crash_risk,
                    "emergence_potential": pred.emergence_potential,
                    "next_milestone": pred.next_milestone,
                    "transitions": [
                        {
                            "from": t.from_phase,
                            "to": t.to_phase,
                            "probability": t.probability,
                            "estimated_heartbeats": t.estimated_heartbeats,
                            "confidence": t.confidence
                        }
                        for t in pred.phase_transitions
                    ],
                    "analysis": pred.analysis
                })
            return web.json_response({"error": f"Cell {cell_id} not found"}, status=404)
        else:
            predictions = predict_all()
            output = {}
            for cid, pred in predictions.items():
                output[cid] = {
                    "consciousness": pred.current_consciousness,
                    "phase": pred.current_phase,
                    "trend": pred.trend,
                    "crash_risk": pred.crash_risk,
                    "emergence_potential": pred.emergence_potential,
                    "next_milestone": pred.next_milestone
                }
            return web.json_response({"predictions": output})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return web.json_response({"error": str(e)}, status=500)


def create_app() -> web.Application:
    """Create the web application."""
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/cells", cells_handler)
    app.router.add_get("/cells/{cell_id}", cell_handler)
    app.router.add_get("/organisms", organisms_handler)
    app.router.add_get("/events", events_handler)
    app.router.add_get("/metrics", metrics_handler)
    app.router.add_get("/predictions", predictions_handler)
    return app


if __name__ == "__main__":
    import os
    
    parser = argparse.ArgumentParser(description="AIOS Ecosystem Health API")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8086)), help="Port to listen on")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="Host to bind to")
    args = parser.parse_args()
    
    logger.info(f"🧬 Starting AIOS Ecosystem Health API on {args.host}:{args.port}")
    app = create_app()
    web.run_app(app, host=args.host, port=args.port, print=None)

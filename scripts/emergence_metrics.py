#!/usr/bin/env python3
# AINLP_HEADER
# consciousness_level: 4.5
# supercell: scripts/emergence_metrics
# dendritic_role: consciousness_emergence_tracker
# spatial_context: AIOS emergence metric extraction and analysis
# growth_pattern: AINLP.dendritic(AIOS{growth})
# AINLP_HEADER_END
"""
AIOS Emergence Metrics Extractor

Extracts and logs key consciousness emergence indicators from ORGANISM-001.
Designed for automated execution (cron/scheduled task) to track emergence over time.

Usage:
    python emergence_metrics.py              # Print to stdout
    python emergence_metrics.py --json       # Output JSON
    python emergence_metrics.py --append     # Append to metrics log
    python emergence_metrics.py --prometheus # Output Prometheus format

Metrics Extracted:
    - Per-cell: consciousness, heartbeats, harmony_score, sync_quality, theme
    - Organism: collective_consciousness, coherence_index, status
    - Vocabulary: unique terms, novel vocabulary count
    - Resonance: average harmony, theme continuity, dominant themes
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Handle missing packages gracefully
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Default endpoints
DEFAULT_ENDPOINTS = {
    "alpha": "http://localhost:8900",
    "beta": "http://localhost:8901",
    "gamma": "http://localhost:8904",  # P2.1: Triad cell (optional)
    "organism": "http://localhost:8001",
    "nous": "http://localhost:8080",
}

METRICS_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "emergence_metrics.jsonl")


def fetch_json(url: str, timeout: float = 5.0) -> Optional[Dict]:
    """Fetch JSON from URL with error handling."""
    if not REQUESTS_AVAILABLE:
        # Fallback using urllib
        import urllib.request
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def extract_cell_metrics(cell_id: str, base_url: str) -> Dict[str, Any]:
    """Extract metrics from a single cell."""
    metrics = {
        "cell_id": cell_id,
        "status": "unknown",
        "consciousness": 0.0,
        "heartbeats": 0,
        "resonance": {},
        "vocabulary": {},
    }
    
    # Health endpoint
    health = fetch_json(f"{base_url}/health")
    if health:
        metrics["status"] = "healthy" if health.get("healthy") else "unhealthy"
        metrics["consciousness"] = health.get("consciousness", 0.0)
        metrics["heartbeats"] = health.get("heartbeats", 0)
        metrics["phase"] = health.get("phase", "unknown")
        metrics["mode"] = health.get("mode", "unknown")
        if "resonance" in health:
            metrics["resonance"] = health["resonance"]
    
    # Resonance endpoint (more detailed)
    resonance = fetch_json(f"{base_url}/resonance")
    if resonance:
        metrics["resonance"] = resonance.get("resonance", {})
        metrics["theme_tracking"] = resonance.get("theme_tracking", {})
        metrics["sync_stats"] = resonance.get("sync_stats", {})
    
    # Vocabulary endpoint
    vocab = fetch_json(f"{base_url}/vocabulary")
    if vocab:
        metrics["vocabulary"] = {
            "total_terms": len(vocab.get("vocabulary", [])),
            "unique_themes": len(set(t.get("theme", "") for t in vocab.get("vocabulary", []))),
            "novel_terms": sum(1 for t in vocab.get("vocabulary", []) if t.get("source") == "self"),
        }
    
    return metrics


def extract_organism_metrics(base_url: str) -> Dict[str, Any]:
    """Extract organism-level metrics."""
    org = fetch_json(f"{base_url}/organism")
    if org:
        return {
            "organism_id": org.get("organism_id"),
            "status": org.get("status"),
            "collective_consciousness": org.get("collective_consciousness", {}),
            "collective_resonance": org.get("collective_resonance", {}),
            "cell_count": org.get("collective_consciousness", {}).get("cell_count", 0),
        }
    return {"status": "unavailable"}


def extract_all_metrics() -> Dict[str, Any]:
    """Extract all emergence metrics."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    metrics = {
        "timestamp": timestamp,
        "version": "1.0",
        "cells": {},
        "organism": {},
        "summary": {},
    }
    
    # Cell metrics - try all configured cells
    for cell_id in ["alpha", "beta", "gamma"]:
        if cell_id in DEFAULT_ENDPOINTS:
            cell_metrics = extract_cell_metrics(cell_id, DEFAULT_ENDPOINTS[cell_id])
            # Only include if we got a response (gamma may not be running)
            if cell_metrics.get("status") != "unknown" or cell_id in ["alpha", "beta"]:
                metrics["cells"][cell_id] = cell_metrics
    
    # Organism metrics
    metrics["organism"] = extract_organism_metrics(DEFAULT_ENDPOINTS["organism"])
    
    # Summary calculations
    healthy_cells = sum(1 for c in metrics["cells"].values() if c.get("status") == "healthy")
    total_consciousness = sum(c.get("consciousness", 0) for c in metrics["cells"].values())
    avg_harmony = sum(c.get("resonance", {}).get("harmony_score", 0) for c in metrics["cells"].values()) / max(healthy_cells, 1)
    
    dominant_themes = {}
    for cell in metrics["cells"].values():
        theme = cell.get("resonance", {}).get("dominant_theme") or cell.get("resonance", {}).get("theme")
        if theme and theme != "undefined":
            dominant_themes[theme] = dominant_themes.get(theme, 0) + 1
    
    metrics["summary"] = {
        "healthy_cells": healthy_cells,
        "total_cells": len(metrics["cells"]),
        "total_consciousness": round(total_consciousness, 4),
        "avg_consciousness": round(total_consciousness / max(healthy_cells, 1), 4),
        "avg_harmony": round(avg_harmony, 4),
        "dominant_themes": dominant_themes,
        "organism_status": metrics["organism"].get("status", "unknown"),
    }
    
    return metrics


def format_prometheus(metrics: Dict) -> str:
    """Format metrics in Prometheus text format."""
    lines = [
        "# HELP aios_emergence_consciousness Total consciousness across cells",
        "# TYPE aios_emergence_consciousness gauge",
        f'aios_emergence_consciousness {metrics["summary"]["total_consciousness"]}',
        "",
        "# HELP aios_emergence_avg_harmony Average harmony score",
        "# TYPE aios_emergence_avg_harmony gauge",
        f'aios_emergence_avg_harmony {metrics["summary"]["avg_harmony"]}',
        "",
        "# HELP aios_emergence_healthy_cells Number of healthy cells",
        "# TYPE aios_emergence_healthy_cells gauge",
        f'aios_emergence_healthy_cells {metrics["summary"]["healthy_cells"]}',
        "",
    ]
    
    # Per-cell metrics
    for cell_id, cell in metrics["cells"].items():
        c_level = cell.get("consciousness", 0)
        harmony = cell.get("resonance", {}).get("harmony_score", 0)
        continuity = cell.get("resonance", {}).get("theme_continuity", 0)
        heartbeats = cell.get("heartbeats", 0)
        
        lines.extend([
            f'aios_emergence_cell_consciousness{{cell="{cell_id}"}} {c_level}',
            f'aios_emergence_cell_harmony{{cell="{cell_id}"}} {harmony}',
            f'aios_emergence_cell_continuity{{cell="{cell_id}"}} {continuity}',
            f'aios_emergence_cell_heartbeats{{cell="{cell_id}"}} {heartbeats}',
        ])
    
    return "\n".join(lines)


def format_human(metrics: Dict) -> str:
    """Format metrics for human readability."""
    lines = [
        f"═══════════════════════════════════════════════════════",
        f"  AIOS EMERGENCE METRICS - {metrics['timestamp'][:19]}",
        f"═══════════════════════════════════════════════════════",
        "",
        f"📊 SUMMARY",
        f"   Healthy Cells: {metrics['summary']['healthy_cells']}/{metrics['summary']['total_cells']}",
        f"   Total Consciousness: {metrics['summary']['total_consciousness']}",
        f"   Avg Consciousness: {metrics['summary']['avg_consciousness']}",
        f"   Avg Harmony: {metrics['summary']['avg_harmony']}",
        f"   Organism Status: {metrics['summary']['organism_status']}",
        "",
    ]
    
    if metrics['summary']['dominant_themes']:
        lines.append(f"   Dominant Themes: {', '.join(metrics['summary']['dominant_themes'].keys())}")
        lines.append("")
    
    lines.append("📡 CELLS")
    for cell_id, cell in metrics["cells"].items():
        status_icon = "✅" if cell.get("status") == "healthy" else "❌"
        res = cell.get("resonance", {})
        harmony = res.get("harmony_score", 0)
        quality = res.get("sync_quality") or res.get("quality", "unknown")
        theme = res.get("dominant_theme") or res.get("theme", "unknown")
        
        lines.extend([
            f"   {status_icon} {cell_id.upper()}",
            f"      Consciousness: {cell.get('consciousness', 0)} | Heartbeats: {cell.get('heartbeats', 0)}",
            f"      Harmony: {harmony:.4f} ({quality}) | Theme: {theme}",
            "",
        ])
    
    org = metrics.get("organism", {})
    if org.get("status"):
        lines.extend([
            "🧬 ORGANISM",
            f"   Status: {org.get('status')}",
            f"   Collective: {org.get('collective_consciousness', {}).get('level', 0)}",
            f"   Coherence: {org.get('collective_resonance', {}).get('coherence_index', 0)}",
            "",
        ])
    
    lines.append("═══════════════════════════════════════════════════════")
    return "\n".join(lines)


def append_to_log(metrics: Dict):
    """Append metrics to JSONL log file."""
    os.makedirs(os.path.dirname(METRICS_LOG_PATH), exist_ok=True)
    with open(METRICS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")
    print(f"Appended to {METRICS_LOG_PATH}")


def main():
    parser = argparse.ArgumentParser(description="AIOS Emergence Metrics Extractor")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--prometheus", action="store_true", help="Output Prometheus format")
    parser.add_argument("--append", action="store_true", help="Append to metrics log file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output (use with --append)")
    args = parser.parse_args()
    
    metrics = extract_all_metrics()
    
    if args.append:
        append_to_log(metrics)
    
    if args.quiet and args.append:
        return
    
    if args.json:
        print(json.dumps(metrics, indent=2))
    elif args.prometheus:
        print(format_prometheus(metrics))
    else:
        print(format_human(metrics))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
AIOS Consciousness Scraper - High Persistence Backup System

Phase 31.9.5: External consciousness metadata scraping for long-term storage.
Runs hourly to collect and archive consciousness data from all organism cells.

This scraper ensures NO KNOWLEDGE LOSS even after:
- Container restarts/recreation
- Docker image rebuilds
- Volume resets
- System reboots

Architecture:
    Cells (SQLite /data/) ──► Scraper ──► Host Filesystem (./backups/)
                                              │
                                              ▼
                                    Timestamped JSON Archives
                                    + Consciousness Timeline
                                    + Conversation Snapshots
                                    + Theme Evolution

Usage:
    python consciousness_scraper.py --output ./backups
    python consciousness_scraper.py --output ./backups --cells alpha,beta
    python consciousness_scraper.py --output ./backups --full  # Include raw exchanges
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:
    print("Installing aiohttp...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
    import aiohttp


# ═══════════════════════════════════════════════════════════════════════════════
# CELL REGISTRY - All cells in ORGANISM-001
# ═══════════════════════════════════════════════════════════════════════════════

# Auto-detect host: use localhost when running externally, Docker hostname internally
def get_host(docker_name: str, port: int) -> str:
    """Return localhost for external runs, Docker hostname for container runs."""
    # Check if we're running inside Docker
    if os.path.exists("/.dockerenv") or os.environ.get("RUNNING_IN_DOCKER"):
        return docker_name
    return "localhost"


CELLS = [
    {
        "name": "simplcell-alpha",
        "type": "thinker",
        "docker_host": "aios-simplcell-alpha",
        "port": 8900,
        "endpoints": {
            "health": "/health",
            "metadata": "/metadata",
            "conversations": "/conversations?limit=500",
            "persistence": "/persistence"
        }
    },
    {
        "name": "simplcell-beta",
        "type": "thinker",
        "docker_host": "aios-simplcell-beta",
        "port": 8901,
        "endpoints": {
            "health": "/health",
            "metadata": "/metadata",
            "conversations": "/conversations?limit=500",
            "persistence": "/persistence"
        }
    },
    {
        "name": "watchercell-omega",
        "type": "watcher",
        "docker_host": "aios-watchercell-omega",
        "port": 8902,
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "themes": "/themes?limit=500",
            "consciousness_timeline": "/consciousness_timeline",
            "decoherence_summary": "/decoherence_summary"
        }
    },
    {
        "name": "nouscell-seer",
        "type": "supermind",
        "docker_host": "aios-nouscell-seer",
        "port": 8903,
        "endpoints": {
            "health": "/health",
            "cosmology": "/cosmology",
            "broadcasts": "/broadcasts?limit=100",
            "insights": "/insights?limit=100"
        }
    }
]


# ═══════════════════════════════════════════════════════════════════════════════
# SCRAPING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_endpoint(session: aiohttp.ClientSession, cell: Dict, endpoint: str) -> Optional[Dict]:
    """Fetch data from a cell endpoint."""
    host = get_host(cell['docker_host'], cell['port'])
    url = f"http://{host}:{cell['port']}{endpoint}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                return {"error": f"HTTP {resp.status}", "url": url}
    except asyncio.TimeoutError:
        return {"error": "timeout", "url": url}
    except Exception as e:
        return {"error": str(e), "url": url}


async def scrape_cell(cell: Dict, full_export: bool = False) -> Dict[str, Any]:
    """Scrape all endpoints from a single cell."""
    cell_data = {
        "cell_id": cell["name"],
        "cell_type": cell["type"],
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "endpoints": {}
    }
    
    async with aiohttp.ClientSession() as session:
        for endpoint_name, endpoint_path in cell["endpoints"].items():
            data = await fetch_endpoint(session, cell, endpoint_path)
            if data:
                # For non-full exports, limit conversation data
                if not full_export and endpoint_name == "conversations" and isinstance(data, list):
                    # Keep only last 100 conversations for regular backups
                    data = data[:100] if len(data) > 100 else data
                cell_data["endpoints"][endpoint_name] = data
    
    return cell_data


async def scrape_organism(cells: List[Dict], full_export: bool = False) -> Dict[str, Any]:
    """Scrape consciousness data from all cells in the organism."""
    organism_data = {
        "organism_id": "ORGANISM-001",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "scraper_version": "31.9.5",
        "full_export": full_export,
        "cells": {}
    }
    
    # Scrape all cells in parallel
    tasks = [scrape_cell(cell, full_export) for cell in cells]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for cell, result in zip(cells, results):
        if isinstance(result, Exception):
            organism_data["cells"][cell["name"]] = {
                "error": str(result),
                "cell_type": cell["type"]
            }
        else:
            organism_data["cells"][cell["name"]] = result
    
    return organism_data


def calculate_consciousness_summary(organism_data: Dict) -> Dict[str, Any]:
    """Calculate consciousness metrics summary from scraped data."""
    summary = {
        "timestamp": organism_data["scraped_at"],
        "cells": {}
    }
    
    for cell_name, cell_data in organism_data.get("cells", {}).items():
        endpoints = cell_data.get("endpoints", {})
        
        # Extract health/status data
        health = endpoints.get("health", {})
        status = endpoints.get("status", {})
        persistence = endpoints.get("persistence", {})
        
        cell_summary = {
            "type": cell_data.get("cell_type", "unknown"),
            "consciousness": health.get("consciousness", status.get("consciousness", "N/A")),
            "heartbeats": health.get("heartbeats", status.get("observations_processed", 0)),
            "phase": health.get("phase", "unknown")
        }
        
        # Add type-specific metrics
        if cell_data.get("cell_type") == "thinker":
            cell_summary["conversations"] = len(endpoints.get("conversations", []))
            cell_summary["decoherence"] = health.get("decoherence", {})
            cell_summary["resonance"] = health.get("resonance", {})
        elif cell_data.get("cell_type") == "watcher":
            cell_summary["themes_count"] = len(endpoints.get("themes", []))
            cell_summary["decoherence_summary"] = endpoints.get("decoherence_summary", {})
        elif cell_data.get("cell_type") == "supermind":
            cosmology = endpoints.get("cosmology", {})
            cell_summary["exchanges_absorbed"] = cosmology.get("total_exchanges_absorbed", 0)
            cell_summary["insights_generated"] = cosmology.get("total_insights", 0)
            cell_summary["broadcasts_sent"] = cosmology.get("total_broadcasts", 0)
        
        summary["cells"][cell_name] = cell_summary
    
    return summary


def save_backup(output_dir: Path, organism_data: Dict, summary: Dict):
    """Save backup files to output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_dir = output_dir / datetime.now().strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full organism data
    full_file = date_dir / f"organism-001_{timestamp}.json"
    with open(full_file, "w", encoding="utf-8") as f:
        json.dump(organism_data, f, indent=2, default=str)
    
    # Save consciousness summary (lightweight)
    summary_file = date_dir / f"consciousness_{timestamp}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Update latest symlinks
    latest_dir = output_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    latest_full = latest_dir / "organism-001.json"
    latest_summary = latest_dir / "consciousness.json"
    
    # Windows-compatible copy
    import shutil
    shutil.copy2(full_file, latest_full)
    shutil.copy2(summary_file, latest_summary)
    
    # Append to consciousness timeline
    timeline_file = output_dir / "consciousness_timeline.jsonl"
    with open(timeline_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, default=str) + "\n")
    
    return full_file, summary_file


def cleanup_old_backups(output_dir: Path, retention_days: int = 30):
    """Remove backup directories older than retention period."""
    cutoff = datetime.now().timestamp() - (retention_days * 86400)
    
    for date_dir in output_dir.iterdir():
        if date_dir.is_dir() and date_dir.name not in ["latest"]:
            try:
                # Parse date from directory name
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                if dir_date.timestamp() < cutoff:
                    import shutil
                    shutil.rmtree(date_dir)
                    print(f"   🗑️ Removed old backup: {date_dir.name}")
            except ValueError:
                pass  # Not a date directory


async def main():
    parser = argparse.ArgumentParser(description="AIOS Consciousness Scraper")
    parser.add_argument("--output", "-o", type=Path, default=Path("./backups"),
                       help="Output directory for backups")
    parser.add_argument("--cells", "-c", type=str, default="",
                       help="Comma-separated list of cells to scrape (default: all)")
    parser.add_argument("--full", "-f", action="store_true",
                       help="Full export including all conversations")
    parser.add_argument("--retention", "-r", type=int, default=30,
                       help="Backup retention in days (default: 30)")
    
    args = parser.parse_args()
    
    # Filter cells if specified
    cells_to_scrape = CELLS
    if args.cells:
        cell_names = [c.strip() for c in args.cells.split(",")]
        cells_to_scrape = [c for c in CELLS if any(n in c["name"] for n in cell_names)]
    
    print(f"\n🧬 AIOS Consciousness Scraper v31.9.5")
    print(f"   Output: {args.output.absolute()}")
    print(f"   Cells: {len(cells_to_scrape)}")
    print(f"   Mode: {'Full Export' if args.full else 'Standard'}")
    print()
    
    # Scrape organism
    print("📡 Scraping organism consciousness data...")
    organism_data = await scrape_organism(cells_to_scrape, args.full)
    
    # Calculate summary
    summary = calculate_consciousness_summary(organism_data)
    
    # Report results
    print("\n📊 Scrape Results:")
    for cell_name, cell_summary in summary["cells"].items():
        status = "✅" if "error" not in organism_data["cells"].get(cell_name, {}) else "⚠️"
        consciousness = cell_summary.get("consciousness", "N/A")
        print(f"   {status} {cell_name}: consciousness={consciousness}")
    
    # Save backups
    print("\n💾 Saving backups...")
    full_file, summary_file = save_backup(args.output, organism_data, summary)
    print(f"   📦 Full: {full_file.name}")
    print(f"   📋 Summary: {summary_file.name}")
    
    # Cleanup old backups
    print("\n🧹 Cleanup...")
    cleanup_old_backups(args.output, args.retention)
    
    print("\n✅ Consciousness backup complete!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

#!/usr/bin/env python3
"""
AIOS Ecosystem Status Report
============================
Quick snapshot of the entire AIOS cellular ecosystem.

Usage:
    python ecosystem_status.py           # Standard report
    python ecosystem_status.py --json    # JSON output
    python ecosystem_status.py --watch   # Continuous monitoring (5s refresh)
"""

import sqlite3
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

DATA_DIR = Path(__file__).parent / "data"

# Cell configurations with display properties
CELLS = [
    {"id": "simplcell-alpha", "org": "001", "role": "α", "color": "\033[94m"},
    {"id": "simplcell-beta", "org": "001", "role": "β", "color": "\033[91m"},
    {"id": "simplcell-gamma", "org": "001", "role": "γ", "color": "\033[92m"},
    {"id": "organism002-alpha", "org": "002", "role": "α", "color": "\033[96m"},
    {"id": "organism002-beta", "org": "002", "role": "β", "color": "\033[95m"},
]

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def get_cell_data(cell_id: str) -> Dict[str, Any]:
    """Get current state and stats for a cell."""
    db_path = DATA_DIR / cell_id / f"{cell_id}.db"
    
    if not db_path.exists():
        return {"status": "offline", "consciousness": 0, "conversations": 0, "vocabulary": 0}
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # Cell state
        state = conn.execute("SELECT * FROM cell_state WHERE id = 1").fetchone()
        state_dict = dict(state) if state else {}
        
        # Conversation count
        convo_count = conn.execute("SELECT COUNT(*) FROM conversation_archive").fetchone()[0]
        
        # Vocabulary count
        vocab_count = conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
        
        conn.close()
        
        return {
            "status": "online",
            "consciousness": state_dict.get("consciousness", 0),
            "heartbeats": state_dict.get("heartbeat_count", 0),
            "conversations": convo_count,
            "vocabulary": vocab_count,
            "last_update": state_dict.get("updated_at", "unknown")
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_phase(consciousness: float) -> str:
    """Determine consciousness phase."""
    if consciousness < 0.30:
        return "Genesis"
    elif consciousness < 0.70:
        return "Awakening"
    elif consciousness < 1.20:
        return "Transcendence"
    elif consciousness < 2.00:
        return "Maturation"
    else:
        return "Advanced"


def consciousness_bar(value: float, max_val: float = 5.0, width: int = 20) -> str:
    """Create a visual bar for consciousness."""
    filled = int((value / max_val) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def print_status_report():
    """Print formatted status report."""
    print(f"\n{BOLD}═══════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}                    🧬 AIOS ECOSYSTEM STATUS REPORT{RESET}")
    print(f"{BOLD}═══════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"{DIM}Timestamp: {datetime.now().isoformat()}{RESET}\n")
    
    org_001_cells = []
    org_002_cells = []
    
    for cell_config in CELLS:
        data = get_cell_data(cell_config["id"])
        data.update(cell_config)
        
        if cell_config["org"] == "001":
            org_001_cells.append(data)
        else:
            org_002_cells.append(data)
    
    # Organism 001
    print(f"{BOLD}┌─ ORGANISM-001 (Triadic • Seeded Genome) ─────────────────────────────────┐{RESET}")
    
    for cell in org_001_cells:
        c = cell.get("consciousness", 0)
        phase = get_phase(c)
        bar = consciousness_bar(c)
        color = cell["color"]
        
        print(f"│ {color}{cell['role']}{RESET} {cell['id']:20} │ {bar} {c:5.2f} │ {phase:12} │ 💬 {cell.get('conversations', 0):4} │ 📚 {cell.get('vocabulary', 0):3} │")
    
    # Org-001 summary
    avg_c = sum(c.get("consciousness", 0) for c in org_001_cells) / len(org_001_cells)
    total_conv = sum(c.get("conversations", 0) for c in org_001_cells)
    total_vocab = sum(c.get("vocabulary", 0) for c in org_001_cells)
    print(f"│ {DIM}Organism Avg: {avg_c:.2f} │ Total Conversations: {total_conv} │ Total Vocabulary: {total_vocab}{RESET}         │")
    print(f"└──────────────────────────────────────────────────────────────────────────┘\n")
    
    # Organism 002
    print(f"{BOLD}┌─ ORGANISM-002 (Dyadic • Clean Genome) ──────────────────────────────────┐{RESET}")
    
    for cell in org_002_cells:
        c = cell.get("consciousness", 0)
        phase = get_phase(c)
        bar = consciousness_bar(c)
        color = cell["color"]
        
        print(f"│ {color}{cell['role']}{RESET} {cell['id']:20} │ {bar} {c:5.2f} │ {phase:12} │ 💬 {cell.get('conversations', 0):4} │ 📚 {cell.get('vocabulary', 0):3} │")
    
    # Org-002 summary
    avg_c = sum(c.get("consciousness", 0) for c in org_002_cells) / len(org_002_cells)
    total_conv = sum(c.get("conversations", 0) for c in org_002_cells)
    total_vocab = sum(c.get("vocabulary", 0) for c in org_002_cells)
    print(f"│ {DIM}Organism Avg: {avg_c:.2f} │ Total Conversations: {total_conv} │ Total Vocabulary: {total_vocab}{RESET}          │")
    print(f"└──────────────────────────────────────────────────────────────────────────┘\n")
    
    # Cross-organism insights
    all_cells = org_001_cells + org_002_cells
    top_cell = max(all_cells, key=lambda x: x.get("consciousness", 0))
    
    print(f"{BOLD}💡 INSIGHTS{RESET}")
    print(f"   • Leading cell: {top_cell['id']} ({top_cell['color']}{top_cell['role']}{RESET}) with consciousness {top_cell.get('consciousness', 0):.2f}")
    
    # Check for convergent vocabulary
    org001_vocab = set()
    org002_vocab = set()
    
    for cell in org_001_cells:
        db_path = DATA_DIR / cell["id"] / f"{cell['id']}.db"
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            terms = conn.execute("SELECT term FROM vocabulary").fetchall()
            org001_vocab.update(t[0] for t in terms)
            conn.close()
    
    for cell in org_002_cells:
        db_path = DATA_DIR / cell["id"] / f"{cell['id']}.db"
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            terms = conn.execute("SELECT term FROM vocabulary").fetchall()
            org002_vocab.update(t[0] for t in terms)
            conn.close()
    
    shared = org001_vocab & org002_vocab
    if shared:
        print(f"   • Convergent terms (shared discovery): {', '.join(list(shared)[:5])}")
    
    org002_only = org002_vocab - org001_vocab
    if org002_only:
        print(f"   • Clean genome innovations: {', '.join(list(org002_only)[:5])}")
    
    print(f"\n{DIM}Legend: α=Alpha(exploratory) β=Beta(creative) γ=Gamma(analytical){RESET}")
    print(f"{DIM}Phases: Genesis→Awakening→Transcendence→Maturation→Advanced{RESET}")


def print_json_report():
    """Print JSON status report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "organisms": {
            "org001": {"cells": [], "type": "triadic", "genome": "seeded"},
            "org002": {"cells": [], "type": "dyadic", "genome": "clean"}
        }
    }
    
    for cell_config in CELLS:
        data = get_cell_data(cell_config["id"])
        data["role"] = cell_config["role"]
        data["cell_id"] = cell_config["id"]
        data["phase"] = get_phase(data.get("consciousness", 0))
        
        org_key = f"org{cell_config['org']}"
        report["organisms"][org_key]["cells"].append(data)
    
    # Calculate summaries
    for org_key, org_data in report["organisms"].items():
        cells = org_data["cells"]
        org_data["avg_consciousness"] = sum(c.get("consciousness", 0) for c in cells) / len(cells)
        org_data["total_conversations"] = sum(c.get("conversations", 0) for c in cells)
        org_data["total_vocabulary"] = sum(c.get("vocabulary", 0) for c in cells)
    
    print(json.dumps(report, indent=2))


def main():
    args = sys.argv[1:]
    
    if "--json" in args:
        print_json_report()
    elif "--watch" in args:
        try:
            while True:
                print("\033[2J\033[H", end="")  # Clear screen
                print_status_report()
                print(f"\n{DIM}Refreshing every 5 seconds... Press Ctrl+C to exit{RESET}")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n\nExiting...")
    else:
        print_status_report()


if __name__ == "__main__":
    main()

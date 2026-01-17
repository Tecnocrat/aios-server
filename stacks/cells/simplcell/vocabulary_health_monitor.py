#!/usr/bin/env python3
"""
AIOS Vocabulary Health Monitor
==============================
Provides metrics and insights about vocabulary emergence across organisms.
Can be called by external systems (Grafana, health checks) to monitor
linguistic evolution.

Outputs JSON for easy integration with monitoring systems.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"

ORGANISMS = {
    "org001": {
        "name": "Organism-001",
        "cells": ["simplcell-alpha", "simplcell-beta", "simplcell-gamma"],
        "type": "triadic",
        "genome": "seeded"
    },
    "org002": {
        "name": "Organism-002", 
        "cells": ["organism002-alpha", "organism002-beta"],
        "type": "dyadic",
        "genome": "clean"
    }
}

def get_db_path(cell_id: str) -> Path:
    return DATA_DIR / cell_id / f"{cell_id}.db"

def safe_query(db_path: Path, query: str, params: tuple = ()) -> list:
    """Execute query safely, returning empty list on errors."""
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    except:
        return []

def get_organism_metrics(org_id: str) -> dict:
    """Get vocabulary metrics for an organism."""
    config = ORGANISMS.get(org_id, {})
    cells = config.get("cells", [])
    
    total_terms = 0
    unique_terms = set()
    term_details = []
    total_usage = 0
    cell_metrics = {}
    
    for cell_id in cells:
        db_path = get_db_path(cell_id)
        vocab = safe_query(db_path, "SELECT * FROM vocabulary")
        
        cell_term_count = len(vocab)
        cell_usage = sum(v.get("usage_count", 0) for v in vocab)
        
        cell_metrics[cell_id] = {
            "term_count": cell_term_count,
            "total_usage": cell_usage
        }
        
        total_terms += cell_term_count
        total_usage += cell_usage
        
        for v in vocab:
            term = v.get("term", "")
            unique_terms.add(term)
            term_details.append({
                "term": term,
                "cell": cell_id,
                "usage": v.get("usage_count", 0),
                "first_consciousness": v.get("first_seen_consciousness", 0)
            })
    
    # Sort by usage to get top terms
    term_details.sort(key=lambda x: -x["usage"])
    
    return {
        "organism": config.get("name", org_id),
        "topology": config.get("type", "unknown"),
        "genome_type": config.get("genome", "unknown"),
        "metrics": {
            "total_vocabulary_entries": total_terms,
            "unique_terms": len(unique_terms),
            "total_usage_count": total_usage,
            "average_usage_per_term": total_usage / max(len(unique_terms), 1),
            "vocabulary_diversity": len(unique_terms) / max(total_terms, 1)
        },
        "top_terms": term_details[:5],
        "cells": cell_metrics
    }

def get_convergence_metrics() -> dict:
    """Analyze vocabulary convergence between organisms."""
    org001_terms = set()
    org002_terms = set()
    
    for cell_id in ORGANISMS["org001"]["cells"]:
        vocab = safe_query(get_db_path(cell_id), "SELECT term FROM vocabulary")
        org001_terms.update(v["term"] for v in vocab)
    
    for cell_id in ORGANISMS["org002"]["cells"]:
        vocab = safe_query(get_db_path(cell_id), "SELECT term FROM vocabulary")
        org002_terms.update(v["term"] for v in vocab)
    
    shared = org001_terms & org002_terms
    org001_only = org001_terms - org002_terms
    org002_only = org002_terms - org001_terms
    
    # Jaccard similarity for convergence
    jaccard = len(shared) / max(len(org001_terms | org002_terms), 1)
    
    return {
        "convergent_terms": list(shared),
        "convergent_count": len(shared),
        "org001_exclusive": len(org001_only),
        "org002_exclusive": len(org002_only),
        "jaccard_similarity": jaccard,
        "convergence_ratio": len(shared) / max(len(org002_terms), 1) if org002_terms else 0,
        "clean_genome_innovations": list(org002_only)
    }

def get_health_report(format: str = "json") -> Optional[str]:
    """Generate complete vocabulary health report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "organisms": {},
        "cross_organism": None,
        "health_score": 0.0,
        "alerts": []
    }
    
    # Get metrics for each organism
    for org_id in ORGANISMS:
        report["organisms"][org_id] = get_organism_metrics(org_id)
    
    # Get convergence metrics
    report["cross_organism"] = get_convergence_metrics()
    
    # Calculate health score (0-100)
    health_factors = []
    
    # Factor 1: Vocabulary exists
    total_unique = sum(o["metrics"]["unique_terms"] for o in report["organisms"].values())
    health_factors.append(min(total_unique / 100, 1.0) * 30)  # Up to 30 points
    
    # Factor 2: Clean genome is developing vocabulary
    org002_terms = report["organisms"].get("org002", {}).get("metrics", {}).get("unique_terms", 0)
    if org002_terms > 0:
        health_factors.append(min(org002_terms / 10, 1.0) * 25)  # Up to 25 points
    else:
        report["alerts"].append({
            "level": "warning",
            "message": "Clean genome organism has not developed any vocabulary"
        })
    
    # Factor 3: Convergence happening
    convergent = report["cross_organism"].get("convergent_count", 0)
    health_factors.append(min(convergent / 5, 1.0) * 20)  # Up to 20 points
    
    # Factor 4: Vocabulary is being used (not just stored)
    total_usage = sum(o["metrics"]["total_usage_count"] for o in report["organisms"].values())
    health_factors.append(min(total_usage / 1000, 1.0) * 25)  # Up to 25 points
    
    report["health_score"] = round(sum(health_factors), 1)
    
    # Add insights
    report["insights"] = []
    
    if convergent > 0:
        report["insights"].append(f"🔄 {convergent} terms discovered independently by both organisms (convergent evolution)")
    
    innovations = report["cross_organism"].get("clean_genome_innovations", [])
    if innovations:
        report["insights"].append(f"🌱 Clean genome has {len(innovations)} unique innovations: {', '.join(innovations[:5])}")
    
    if format == "json":
        return json.dumps(report, indent=2)
    else:
        return report

def main():
    """CLI entry point."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--prometheus":
        # Output Prometheus metrics format
        report = get_health_report(format="dict")
        print("# HELP aios_vocabulary_health_score Overall vocabulary health score (0-100)")
        print("# TYPE aios_vocabulary_health_score gauge")
        print(f"aios_vocabulary_health_score {report['health_score']}")
        
        for org_id, org_data in report["organisms"].items():
            metrics = org_data["metrics"]
            print(f'aios_vocabulary_unique_terms{{organism="{org_id}"}} {metrics["unique_terms"]}')
            print(f'aios_vocabulary_total_usage{{organism="{org_id}"}} {metrics["total_usage_count"]}')
        
        print(f"aios_vocabulary_convergent_terms {report['cross_organism']['convergent_count']}")
        print(f"aios_vocabulary_jaccard_similarity {report['cross_organism']['jaccard_similarity']:.4f}")
    else:
        # Standard JSON output
        print(get_health_report())

if __name__ == "__main__":
    main()

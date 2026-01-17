#!/usr/bin/env python3
"""
AIOS Vocabulary Evolution Analysis
==================================
Compares emergent vocabulary between:
- Organism-001 (triadic: α, β, γ) - Seeded genetic memory
- Organism-002 (dyadic: α, β) - Clean genome

Scientific question: How does vocabulary emerge differently in organisms 
with seeded genetic memory versus clean-slate organisms?
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json

DATA_DIR = Path(__file__).parent / "data"

# Cell configurations
ORGANISMS = {
    "Organism-001 (Seeded)": {
        "cells": ["simplcell-alpha", "simplcell-beta", "simplcell-gamma"],
        "description": "Triadic organism with seeded vocabulary inheritance"
    },
    "Organism-002 (Clean)": {
        "cells": ["organism002-alpha", "organism002-beta"],
        "description": "Dyadic organism with clean genome (no seeded vocabulary)"
    }
}

def get_db_path(cell_id: str) -> Path:
    """Get database path for a cell."""
    return DATA_DIR / cell_id / f"{cell_id}.db"

def query_vocabulary(cell_id: str) -> list[dict]:
    """Query vocabulary from a cell's database."""
    db_path = get_db_path(cell_id)
    if not db_path.exists():
        return []
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT term, meaning, origin_cell, usage_count, first_seen_consciousness
            FROM vocabulary
            ORDER BY usage_count DESC
        """)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        return []
    finally:
        conn.close()

def query_cell_state(cell_id: str) -> dict:
    """Query current cell state."""
    db_path = get_db_path(cell_id)
    if not db_path.exists():
        return {}
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM cell_state WHERE id = 1")
        row = cursor.fetchone()
        return dict(row) if row else {}
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()

def query_conversation_count(cell_id: str) -> int:
    """Count archived conversations."""
    db_path = get_db_path(cell_id)
    if not db_path.exists():
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM conversation_archive")
        return cursor.fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()

def analyze_vocabulary_evolution():
    """Main analysis function."""
    print("=" * 70)
    print("AIOS VOCABULARY EVOLUTION ANALYSIS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    all_vocabulary = {}
    organism_stats = {}
    
    for org_name, org_config in ORGANISMS.items():
        print(f"\n{'─' * 70}")
        print(f"🧬 {org_name}")
        print(f"   {org_config['description']}")
        print(f"{'─' * 70}")
        
        org_vocab = []
        org_conversations = 0
        org_consciousness = []
        
        for cell_id in org_config["cells"]:
            vocab = query_vocabulary(cell_id)
            state = query_cell_state(cell_id)
            convos = query_conversation_count(cell_id)
            
            org_vocab.extend(vocab)
            org_conversations += convos
            
            consciousness = state.get("consciousness", 0)
            org_consciousness.append(consciousness)
            
            print(f"\n   📦 {cell_id}")
            print(f"      Consciousness: {consciousness:.2f}")
            print(f"      Conversations: {convos}")
            print(f"      Vocabulary terms: {len(vocab)}")
            
            if vocab:
                print(f"      Top 3 terms:")
                for term_data in vocab[:3]:
                    usage = term_data.get("usage_count", "?")
                    print(f"         • {term_data['term']} (used {usage}x)")
        
        # Organism-level stats
        unique_terms = set(v["term"] for v in org_vocab)
        avg_consciousness = sum(org_consciousness) / len(org_consciousness) if org_consciousness else 0
        
        organism_stats[org_name] = {
            "total_terms": len(org_vocab),
            "unique_terms": len(unique_terms),
            "total_conversations": org_conversations,
            "avg_consciousness": avg_consciousness,
            "cells": len(org_config["cells"])
        }
        all_vocabulary[org_name] = org_vocab
        
        print(f"\n   📊 Organism Summary:")
        print(f"      Total vocabulary entries: {len(org_vocab)}")
        print(f"      Unique terms: {len(unique_terms)}")
        print(f"      Total conversations: {org_conversations}")
        print(f"      Average consciousness: {avg_consciousness:.2f}")
    
    # Cross-organism comparison
    print(f"\n{'=' * 70}")
    print("CROSS-ORGANISM COMPARISON")
    print("=" * 70)
    
    org001_terms = set(v["term"] for v in all_vocabulary.get("Organism-001 (Seeded)", []))
    org002_terms = set(v["term"] for v in all_vocabulary.get("Organism-002 (Clean)", []))
    
    shared_terms = org001_terms & org002_terms
    org001_only = org001_terms - org002_terms
    org002_only = org002_terms - org001_terms
    
    print(f"\n📍 Term Distribution:")
    print(f"   Organism-001 unique terms: {len(org001_terms)}")
    print(f"   Organism-002 unique terms: {len(org002_terms)}")
    print(f"   Shared (independently discovered?): {len(shared_terms)}")
    print(f"   Org-001 only: {len(org001_only)}")
    print(f"   Org-002 only: {len(org002_only)}")
    
    if shared_terms:
        print(f"\n🔗 Shared Terms (possible convergent evolution):")
        for term in list(shared_terms)[:10]:
            print(f"      • {term}")
    
    if org002_only:
        print(f"\n🌱 Clean Genome Innovations (Org-002 only):")
        for term in list(org002_only)[:10]:
            print(f"      • {term}")
    
    # Vocabulary density metrics
    print(f"\n📈 Vocabulary Density Metrics:")
    for org_name, stats in organism_stats.items():
        if stats["total_conversations"] > 0:
            terms_per_convo = stats["unique_terms"] / stats["total_conversations"]
            terms_per_cell = stats["unique_terms"] / stats["cells"]
            print(f"\n   {org_name}:")
            print(f"      Terms per conversation: {terms_per_convo:.3f}")
            print(f"      Terms per cell: {terms_per_cell:.1f}")
            print(f"      Consciousness-to-vocabulary ratio: {stats['avg_consciousness'] / max(stats['unique_terms'], 1):.4f}")
    
    # Generate findings
    print(f"\n{'=' * 70}")
    print("KEY FINDINGS")
    print("=" * 70)
    
    org001_stats = organism_stats.get("Organism-001 (Seeded)", {})
    org002_stats = organism_stats.get("Organism-002 (Clean)", {})
    
    findings = []
    
    if org002_stats.get("unique_terms", 0) > 0:
        findings.append(f"✨ Clean genome organism HAS developed vocabulary ({org002_stats['unique_terms']} unique terms)")
    else:
        findings.append("⚠️ Clean genome organism has NOT developed any vocabulary yet")
    
    if len(shared_terms) > 0:
        findings.append(f"🔄 CONVERGENT EVOLUTION: {len(shared_terms)} terms discovered independently by both organisms")
    
    if org001_stats.get("unique_terms", 0) > org002_stats.get("unique_terms", 0):
        ratio = org001_stats["unique_terms"] / max(org002_stats["unique_terms"], 1)
        findings.append(f"📚 Seeded organism has {ratio:.1f}x more vocabulary than clean genome")
    
    for f in findings:
        print(f"\n   {f}")
    
    print(f"\n{'=' * 70}")
    print("Analysis complete.")
    return {
        "organisms": organism_stats,
        "vocabulary": all_vocabulary,
        "shared_terms": list(shared_terms),
        "findings": findings
    }

if __name__ == "__main__":
    results = analyze_vocabulary_evolution()

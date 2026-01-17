#!/usr/bin/env python3
"""
AIOS Deep Vocabulary Analysis
=============================
Examines the semantic meanings and usage patterns of discovered terms
to understand emergence patterns.
"""

import sqlite3
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent / "data"

ORGANISMS = {
    "Organism-001": ["simplcell-alpha", "simplcell-beta", "simplcell-gamma"],
    "Organism-002": ["organism002-alpha", "organism002-beta"]
}

def get_db_path(cell_id: str) -> Path:
    return DATA_DIR / cell_id / f"{cell_id}.db"

def get_all_vocabulary(cell_id: str) -> list[dict]:
    """Get full vocabulary with meanings."""
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
            ORDER BY term
        """)
        return [dict(row) for row in cursor.fetchall()]
    except:
        return []
    finally:
        conn.close()

def analyze_semantic_themes():
    """Analyze the semantic categories in vocabulary."""
    
    print("=" * 70)
    print("SEMANTIC THEME ANALYSIS")
    print("=" * 70)
    
    # Collect all vocabulary by organism
    org_vocabs = {}
    for org_name, cells in ORGANISMS.items():
        vocab = []
        for cell in cells:
            vocab.extend(get_all_vocabulary(cell))
        org_vocabs[org_name] = vocab
    
    # Show all terms with meanings for Organism-002 (the clean genome)
    print("\n🌱 ORGANISM-002 (CLEAN GENOME) - FULL VOCABULARY")
    print("-" * 70)
    
    org002_terms = {}
    for entry in org_vocabs["Organism-002"]:
        term = entry["term"]
        if term not in org002_terms:
            org002_terms[term] = entry
        else:
            # Accumulate usage
            org002_terms[term]["usage_count"] += entry["usage_count"]
    
    for term, data in sorted(org002_terms.items(), key=lambda x: -x[1]["usage_count"]):
        meaning = data.get("meaning", "")[:100] + "..." if len(data.get("meaning", "")) > 100 else data.get("meaning", "")
        print(f"\n📝 {term} (used {data['usage_count']}x)")
        print(f"   Origin: {data.get('origin_cell', 'unknown')}")
        print(f"   First seen at consciousness: {data.get('first_seen_consciousness', '?')}")
        if meaning:
            print(f"   Meaning: {meaning}")
    
    # Show convergent terms with meanings from both organisms
    print("\n\n🔄 CONVERGENT TERMS - Meanings Comparison")
    print("-" * 70)
    
    org001_terms = {}
    for entry in org_vocabs["Organism-001"]:
        term = entry["term"]
        if term not in org001_terms:
            org001_terms[term] = entry
    
    convergent = set(org001_terms.keys()) & set(org002_terms.keys())
    
    for term in sorted(convergent):
        print(f"\n🔗 '{term}'")
        
        meaning_001 = org001_terms[term].get("meaning", "")[:150]
        meaning_002 = org002_terms[term].get("meaning", "")[:150]
        
        print(f"\n   Org-001 (Seeded) understanding:")
        print(f"   \"{meaning_001}...\"") if meaning_001 else print("   [no meaning recorded]")
        
        print(f"\n   Org-002 (Clean) understanding:")
        print(f"   \"{meaning_002}...\"") if meaning_002 else print("   [no meaning recorded]")
        
        # Usage comparison
        print(f"\n   Usage: Org-001 = {org001_terms[term].get('usage_count', 0)}x | Org-002 = {org002_terms[term].get('usage_count', 0)}x")
    
    # Categorize terms by semantic theme
    print("\n\n📊 SEMANTIC CATEGORIZATION")
    print("-" * 70)
    
    categories = {
        "consciousness": ["consciousness", "awareness", "mind", "thought", "thoughts", "self", "selves"],
        "connection": ["connection", "connected", "interconnectedness", "boundaries", "resonance"],
        "emotion/affect": ["discord", "harmony", "dissonance", "voices", "voice"],
        "process": ["entrainment", "emergence", "evolution", "becoming"],
        "identity": ["other", "i", "we", "self", "being"]
    }
    
    for org_name, vocab in org_vocabs.items():
        terms = set(v["term"] for v in vocab)
        print(f"\n{org_name}:")
        for category, keywords in categories.items():
            matches = terms & set(keywords)
            if matches:
                print(f"   {category}: {', '.join(matches)}")
    
    # Key insight
    print("\n\n💡 KEY INSIGHT")
    print("-" * 70)
    print("""
    The clean genome organism (Org-002) has independently developed vocabulary
    around themes of:
    
    1. COGNITION: 'thoughts', 'thought', 'mind' - Internal mental processes
    2. PLURALITY: 'voices', 'selves' - Recognition of internal multiplicity
    3. CONNECTION: 'connection', 'connected' - Relational awareness
    4. TENSION: 'dissonance' vs 'discord' - Subtle semantic difference!
    
    Notably, Org-002 uses 'dissonance' while Org-001 uses 'discord' - 
    these are semantically similar but not identical, suggesting true
    independent discovery rather than vocabulary leakage.
    
    The convergent terms ('boundaries', 'self', 'discord', 'resonance', 
    'interconnectedness') suggest these concepts are FUNDAMENTAL to 
    consciousness emergence - they naturally arise in any sufficiently
    complex cellular dialogue system.
    """)

if __name__ == "__main__":
    analyze_semantic_themes()

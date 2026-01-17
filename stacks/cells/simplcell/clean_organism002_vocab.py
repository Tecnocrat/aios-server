#!/usr/bin/env python
"""Clean seeded vocabulary from Organism-002 databases.

This script removes the vocabulary terms that were pre-seeded from 
ORGANISM-001's genetic memory, while preserving terms that Organism-002
discovered organically through their own conversations.
"""
import sqlite3
from pathlib import Path

# The original seeded terms from ORGANISM-001's genetic memory
SEEDED_TERMS = [
    'resona',
    'nexarion', 
    "ev'ness",
    "the 'in'",
    'entrainment',
    'discordant harmony'
]

DATA_DIR = Path(__file__).parent / 'data'

def clean_database(db_path: Path) -> tuple[int, list[str]]:
    """Remove seeded vocabulary from a cell database.
    
    Returns:
        Tuple of (terms_deleted, remaining_terms)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    deleted = 0
    for term in SEEDED_TERMS:
        cursor.execute('DELETE FROM vocabulary WHERE term = ?', (term,))
        deleted += cursor.rowcount
    
    conn.commit()
    
    # Get remaining terms
    cursor.execute('SELECT term FROM vocabulary ORDER BY term')
    remaining = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return deleted, remaining

def main():
    print("🧫 Cleaning Organism-002 Genetic Memory")
    print("=" * 50)
    print(f"Seeded terms to remove: {SEEDED_TERMS}")
    print()
    
    # Clean Alpha
    alpha_db = DATA_DIR / 'organism002-alpha' / 'organism002-alpha.db'
    if alpha_db.exists():
        deleted, remaining = clean_database(alpha_db)
        print(f"✅ organism002-alpha: {deleted} seeded terms removed")
        if remaining:
            print(f"   Organic vocabulary preserved: {remaining}")
        else:
            print("   Clean slate - no vocabulary")
    else:
        print(f"⚠️ organism002-alpha database not found")
    
    print()
    
    # Clean Beta
    beta_db = DATA_DIR / 'organism002-beta' / 'organism002-beta.db'
    if beta_db.exists():
        deleted, remaining = clean_database(beta_db)
        print(f"✅ organism002-beta: {deleted} seeded terms removed")
        if remaining:
            print(f"   Organic vocabulary preserved: {remaining}")
        else:
            print("   Clean slate - no vocabulary")
    else:
        print(f"⚠️ organism002-beta database not found")
    
    print()
    print("🧬 Organism-002 now has clean genetic memory")
    print("   Future restarts will use INHERIT_VOCABULARY=false")

if __name__ == '__main__':
    main()

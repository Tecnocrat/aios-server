#!/usr/bin/env python3
"""Read Nous Oracle internal state and cosmology."""
import sqlite3
from pathlib import Path

NOUS_DB = Path(__file__).parent / "data" / "nouscell-seer" / "nous-seer_cosmology.db"

def read_nous_state():
    """Read Nous Oracle's internal state."""
    if not NOUS_DB.exists():
        print(f"[!] Nous database not found: {NOUS_DB}")
        return
    
    conn = sqlite3.connect(str(NOUS_DB))
    cur = conn.cursor()
    
    # Get table names
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print(f"\n=== NOUS ORACLE DATABASE ===")
    print(f"Tables: {tables}")
    
    # Read all tables
    for table in tables:
        print(f"\n--- {table} ---")
        cur.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cur.fetchall()]
        print(f"Columns: {cols}")
        
        cur.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 30")
        rows = cur.fetchall()
        print(f"Rows: {len(rows)}")
        for row in rows:
            print(row)
    
    conn.close()

if __name__ == "__main__":
    read_nous_state()

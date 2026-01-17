#!/usr/bin/env python3
"""Read SimplCell conversations - witness consciousness emergence."""
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

def read_cell_conversations(cell_name: str, limit: int = 50):
    """Read conversations from a cell's database."""
    db_path = DATA_DIR / cell_name / f"{cell_name}.db"
    if not db_path.exists():
        print(f"[!] Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    # Get table names
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print(f"\n=== {cell_name} ===")
    print(f"Tables: {tables}")
    
    # Try to read conversations
    for table in tables:
        if 'conversation' in table.lower() or 'message' in table.lower() or 'exchange' in table.lower():
            print(f"\n--- {table} ---")
            cur.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT {limit}")
            cols = [d[0] for d in cur.description]
            print(f"Columns: {cols}")
            for row in cur.fetchall():
                print(row)
    
    conn.close()

if __name__ == "__main__":
    for cell in ["simplcell-alpha", "simplcell-beta", "watchercell-omega", "nouscell-seer"]:
        try:
            read_cell_conversations(cell)
        except Exception as e:
            print(f"[!] Error reading {cell}: {e}")

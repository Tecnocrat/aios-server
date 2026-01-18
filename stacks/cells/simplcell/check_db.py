import sqlite3
from pathlib import Path

db_path = Path("data/simplcell-gamma/simplcell-gamma.db")
if db_path.exists():
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print(f"Tables: {tables}")
    
    # Get schema for vocabulary-like table
    for table in tables:
        if 'vocab' in table.lower() or table == 'vocabulary':
            cur.execute(f"PRAGMA table_info({table})")
            print(f"\n{table} columns:")
            for col in cur.fetchall():
                print(f"  {col[1]} ({col[2]})")
    
    # Try a query
    if 'vocabulary' in tables:
        cur.execute("SELECT * FROM vocabulary LIMIT 3")
        print(f"\nSample vocabulary:")
        for row in cur.fetchall():
            print(f"  {row}")
    
    conn.close()
else:
    print(f"DB not found at {db_path}")

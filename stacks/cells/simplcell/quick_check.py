#!/usr/bin/env python3
"""Quick comparison of stored vs absorbed consciousness."""
import sqlite3

# Check Alpha's current stored consciousness
conn = sqlite3.connect('data/simplcell-alpha/simplcell-alpha.db')
cursor = conn.cursor()
cursor.execute('SELECT consciousness FROM cell_state WHERE id = 1')
stored = cursor.fetchone()[0]
print(f'Alpha stored consciousness: {stored:.3f}')
conn.close()

# Check what Nous recorded for latest
conn = sqlite3.connect('data/nouscell-seer/nous-seer_cosmology.db')
cursor = conn.cursor()
cursor.execute('''
    SELECT consciousness, absorbed_at
    FROM exchanges
    WHERE source_cell = 'simplcell-alpha'
    ORDER BY absorbed_at DESC
    LIMIT 1
''')
row = cursor.fetchone()
print(f'Nous last absorbed consciousness: {row[0]:.3f} at {row[1]}')
conn.close()

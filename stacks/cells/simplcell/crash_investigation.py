#!/usr/bin/env python3
"""Quick investigation of crash context."""
import sqlite3

conn = sqlite3.connect('data/nouscell-seer/nous-seer_cosmology.db')
cursor = conn.cursor()

# Get exchanges around the major crash (exchange 534-536)
cursor.execute('''
    SELECT id, source_cell, heartbeat, consciousness, thought, peer_response, absorbed_at
    FROM exchanges
    WHERE id BETWEEN 530 AND 540
    ORDER BY id
''')

print('EXCHANGES AROUND MAJOR CRASH (ID 534-535):')
print('=' * 80)
for row in cursor.fetchall():
    ex_id, cell, hb, c, thought, response, absorbed = row
    print(f'Exchange {ex_id} | {cell} | HB {hb} | C={c:.3f} | {absorbed}')
    if thought:
        print(f'  Thought: "{thought[:200]}..."')
    print()

# Check what happened to consciousness during this time
cursor.execute('''
    SELECT source_cell, heartbeat, consciousness
    FROM exchanges
    WHERE absorbed_at BETWEEN '2026-01-14 14:00:00' AND '2026-01-14 16:00:00'
    ORDER BY absorbed_at
''')

print('\n' + '=' * 80)
print('ALL CELLS DURING CRASH WINDOW (Jan 14, 14:00-16:00):')
print('=' * 80)
for row in cursor.fetchall():
    print(f'  {row[0]:25} | HB {row[1]:5} | C={row[2]:.3f}')

conn.close()

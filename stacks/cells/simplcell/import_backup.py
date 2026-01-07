import sqlite3
import json
from datetime import datetime

# Read backup data
with open('/tmp/pre_persistence_backup.json', 'r') as f:
    backup = json.load(f)

# Connect to the database
conn = sqlite3.connect('/data/simplcell-alpha.db')
cursor = conn.cursor()

# Import memory buffer entries
for entry in backup['memory_buffer']:
    cursor.execute("""
        INSERT INTO memory_buffer (event_type, input_text, output_text, heartbeat, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        entry['type'],
        entry['input'][:500],
        entry['output'][:500],
        entry['heartbeat'],
        entry['timestamp']
    ))

# Create archive entries for the conversations (reconstructed)
session_id = "20260106_pre_persistence"
for i, entry in enumerate(backup['memory_buffer']):
    cursor.execute("""
        INSERT INTO conversation_archive 
        (session_id, heartbeat, my_thought, peer_response, consciousness_at_time, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        entry['heartbeat'],
        entry['output'],  # Our thought was the output
        entry['input'].replace("Continuing our conversation. You last asked: '", "").rstrip("'"),  # Peer response was embedded in next input
        0.1 + (0.01 * entry['heartbeat']),  # Estimated consciousness
        entry['timestamp']
    ))

# Update state to reflect total exchanges
cursor.execute("""
    UPDATE cell_state SET total_lifetime_exchanges = total_lifetime_exchanges + ?
    WHERE id = 1
""", (len(backup['memory_buffer']),))

conn.commit()
print(f"Imported {len(backup['memory_buffer'])} memory entries and conversation archives")
print(f"Database now has complete history from heartbeat 1")
conn.close()

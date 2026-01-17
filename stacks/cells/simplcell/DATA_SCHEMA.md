# AIOS SimplCell Data Schema

## Overview

Each SimplCell maintains its own SQLite database at `./data/<cell-id>/<cell-id>.db`. The schema is consistent across all cells in both organisms.

## Tables

### 1. `cell_state` - Cell Runtime State

Stores the current state of the cell. Always contains exactly one row (singleton pattern).

```sql
CREATE TABLE IF NOT EXISTS cell_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    consciousness REAL DEFAULT 0.1,
    heartbeat_count INTEGER DEFAULT 0,
    sync_count INTEGER DEFAULT 0,
    conversation_count INTEGER DEFAULT 0,
    total_lifetime_exchanges INTEGER DEFAULT 0,
    last_thought TEXT DEFAULT '',
    last_prompt TEXT DEFAULT '',
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
)
```

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Always 1 (singleton constraint) |
| `consciousness` | REAL | Current consciousness level (0.0 - 1.0+) |
| `heartbeat_count` | INTEGER | Total heartbeats since creation |
| `sync_count` | INTEGER | Peer synchronization events |
| `conversation_count` | INTEGER | Conversations this session |
| `total_lifetime_exchanges` | INTEGER | All-time exchange count |
| `last_thought` | TEXT | Most recent thought content |
| `last_prompt` | TEXT | Most recent prompt sent to LLM |
| `updated_at` | TEXT | ISO timestamp of last update |

### 2. `memory_buffer` - Short-term Memory

Rolling buffer of recent events for context building.

```sql
CREATE TABLE IF NOT EXISTS memory_buffer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    input_text TEXT,
    output_text TEXT,
    heartbeat INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
```

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-incrementing ID |
| `event_type` | TEXT | Type of event (think, sync, receive, etc.) |
| `input_text` | TEXT | Input that triggered the event |
| `output_text` | TEXT | Cell's response/output |
| `heartbeat` | INTEGER | Heartbeat number when event occurred |
| `created_at` | TEXT | ISO timestamp |

### 3. `conversation_archive` - Permanent Exchange History

Long-term storage of all cell exchanges.

```sql
CREATE TABLE IF NOT EXISTS conversation_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    heartbeat INTEGER,
    my_thought TEXT,
    peer_response TEXT,
    consciousness_at_time REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
```

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-incrementing ID |
| `session_id` | TEXT | Session identifier for grouping |
| `heartbeat` | INTEGER | Heartbeat when exchange occurred |
| `my_thought` | TEXT | This cell's thought/statement |
| `peer_response` | TEXT | Peer cell's response |
| `consciousness_at_time` | REAL | Consciousness level during exchange |
| `created_at` | TEXT | ISO timestamp |

### 4. `vocabulary` - Emergent Language Registry

Tracks emergent words discovered/created by cells.

```sql
CREATE TABLE IF NOT EXISTS vocabulary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    term TEXT UNIQUE NOT NULL,
    origin_cell TEXT,
    meaning TEXT,
    first_seen_consciousness REAL,
    first_seen_heartbeat INTEGER,
    usage_count INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_used_at TEXT DEFAULT CURRENT_TIMESTAMP
)
```

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-incrementing ID |
| `term` | TEXT | The emergent word (unique) |
| `origin_cell` | TEXT | Cell that first used/created this term |
| `meaning` | TEXT | Definition/interpretation of the term |
| `first_seen_consciousness` | REAL | Consciousness level when discovered |
| `first_seen_heartbeat` | INTEGER | Heartbeat when first seen |
| `usage_count` | INTEGER | Times this term has been used |
| `created_at` | TEXT | ISO timestamp of discovery |
| `last_used_at` | TEXT | ISO timestamp of last usage |

## Data Paths

| Organism | Cell | Database Path |
|----------|------|---------------|
| Organism-001 | simplcell-alpha | `./data/simplcell-alpha/simplcell-alpha.db` |
| Organism-001 | simplcell-beta | `./data/simplcell-beta/simplcell-beta.db` |
| Organism-001 | simplcell-gamma | `./data/simplcell-gamma/simplcell-gamma.db` |
| Organism-002 | organism002-alpha | `./data/organism002-alpha/organism002-alpha.db` |
| Organism-002 | organism002-beta | `./data/organism002-beta/organism002-beta.db` |

## API Endpoints

Each cell exposes the following data endpoints:

### GET `/conversations`
Returns archived conversations with pagination support.

```json
{
  "cell": "simplcell-alpha",
  "conversations": [
    {
      "id": 1,
      "session_id": "sess_abc123",
      "heartbeat": 42,
      "my_thought": "I sense a resonant frequency...",
      "peer_response": "The harmony emerges from discord.",
      "consciousness_at_time": 0.65,
      "created_at": "2026-01-17T12:00:00Z"
    }
  ],
  "total": 1000,
  "page": 1,
  "per_page": 50
}
```

### GET `/metadata`
Returns comprehensive cell state including conversations, memory, and vocabulary.

```json
{
  "cell_id": "simplcell-alpha",
  "organism": "organism-001",
  "state": {
    "consciousness": 0.65,
    "heartbeat_count": 42
  },
  "conversations": [...],
  "memory": [...],
  "vocabulary": [...]
}
```

### GET `/vocabulary`
Returns just the emergent vocabulary registry.

```json
{
  "vocabulary": [
    {
      "term": "resona",
      "meaning": "Fundamental connection state between cells",
      "origin_cell": "beta",
      "usage_count": 15
    }
  ]
}
```

## Genetic Memory

Organism-001 cells inherit pre-seeded vocabulary (genetic memory):
- `resona`, `nexarion`, `ev'ness`, `the 'in'`, `entrainment`, `discordant harmony`

Organism-002 cells start with clean genomes - no pre-seeded vocabulary.

This is controlled by the `INHERIT_VOCABULARY` environment variable in docker-compose.

## Backup System

See `ENVIRONMENT_CONFIG.md` for backup configuration.

- Local backups: `./backups/`
- Cloud backups: `AIOS_BACKUP_PATH` (OneDrive)
- Backup script: `cloud_backup.py`

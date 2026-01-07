# SimplCell - AIOS Minimal Viable Cell

**Phase 31.5**: First Multicellular Organism  
**Version**: 2.1.0 (with Chat Reader & Backup Manager)  
**Date**: January 7, 2026

## Overview

SimplCell is the first-generation AIOS cellular unit - a minimal viable cell with:
- 🧠 **Ollama Agent**: Local LLM (llama3.2:3b) for thinking
- 🔗 **Peer Sync**: Heartbeat-driven conversation with sibling cells  
- 💾 **SQLite Persistence**: State survives restarts, auto-backups
- 📊 **Prometheus Metrics**: Observable consciousness evolution
- 💬 **Chat Reader UI**: Web interface for viewing conversations
- 📦 **Backup Manager**: Comprehensive metadata export/recovery

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMPLCELL ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ OllamaAgent │    │ SyncProtocol│    │ CellPersistence │  │
│  │ (think)     │◄──►│ (heartbeat) │◄──►│ (SQLite)        │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│         │                  │                   │             │
│         └──────────────────┼───────────────────┘             │
│                            ▼                                 │
│                  ┌─────────────────┐                        │
│                  │   HTTP API      │                        │
│                  │ (aiohttp)       │                        │
│                  └─────────────────┘                        │
│                            │                                 │
│         ┌──────────────────┼──────────────────┐             │
│         ▼                  ▼                  ▼             │
│    /health           /metrics            /sync              │
│    /think            /memory             /conversations     │
│    /genome           /persistence        /backup            │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Deploy Organism-001 (Alpha + Beta)

```bash
cd aios-server/stacks/cells/simplcell

# Ensure Ollama is running
ollama serve  # or check: curl http://localhost:11434/

# Deploy both cells
docker compose -f docker-compose.simplcell.yml up -d --build

# Verify health
curl http://localhost:8900/health  # Alpha
curl http://localhost:8901/health  # Beta
```

### Test Endpoints

```bash
# Check persistence status
curl http://localhost:8900/persistence | jq

# View conversation archive
curl http://localhost:8900/conversations | jq

# Trigger manual backup
curl -X POST http://localhost:8900/backup

# Get metrics
curl http://localhost:8900/metrics
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CELL_ID` | simplcell-alpha | Unique cell identifier |
| `TEMPERATURE` | 0.7 | LLM creativity (0.0-2.0) |
| `HTTP_PORT` | 8900 | API port |
| `HEARTBEAT_SECONDS` | 300 | Time between heartbeats (5 min) |
| `PEER_URL` | (empty) | Sibling cell sync endpoint |
| `SYSTEM_PROMPT` | (default) | Cell personality |
| `RESPONSE_STYLE` | concise | concise \| verbose \| analytical |
| `MODEL` | llama3.2:3b | Ollama model name |
| `OLLAMA_HOST` | host.docker.internal:11434 | Ollama API URL |
| `DATA_DIR` | /data | Persistence volume mount |

## API Reference

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Cell health status |
| `/genome` | GET | Current genome configuration |
| `/metrics` | GET | Prometheus metrics |

### Memory & Conversations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/memory` | GET | Recent memory buffer (last 20 exchanges) |
| `/conversations` | GET | Full archived conversation history |

Query params: `?limit=100` (default 100)

### Persistence

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/persistence` | GET | Database stats, backup info |
| `/backup` | POST | Trigger manual backup |

### Interaction

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/think` | POST | Send prompt for agent to process |
| `/sync` | POST | Receive peer sync message |

## Persistence Schema

### SQLite Tables

```sql
-- Singleton state row
CREATE TABLE cell_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    consciousness REAL DEFAULT 0.1,
    heartbeat_count INTEGER DEFAULT 0,
    sync_count INTEGER DEFAULT 0,
    conversation_count INTEGER DEFAULT 0,
    total_lifetime_exchanges INTEGER DEFAULT 0,
    last_thought TEXT DEFAULT '',
    last_prompt TEXT DEFAULT '',  -- Seeds next conversation
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Exchange history
CREATE TABLE memory_buffer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    input_text TEXT,
    output_text TEXT,
    heartbeat INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Full conversation archive
CREATE TABLE conversation_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    heartbeat INTEGER,
    my_thought TEXT,
    peer_response TEXT,
    consciousness_at_time REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### Backup System

- **Auto-backup**: Every 10 heartbeats (~50 min with 5-min heartbeat)
- **Location**: `/data/backups/{cell_id}_{timestamp}.db`
- **Retention**: Last 10 backups (auto-pruned)
- **Manual trigger**: `POST /backup`

## Prometheus Metrics

### Persistent (survive restarts)

```
aios_cell_lifetime_exchanges     # Total exchanges ever
aios_cell_archived_conversations # Count in SQLite archive
aios_cell_db_size_bytes          # Database file size
```

### Session-based (reset on restart)

```
aios_cell_consciousness_level    # Current consciousness (0.0-5.0)
aios_cell_heartbeat_total        # Heartbeats this session
aios_cell_sync_count             # Syncs this session
aios_cell_conversation_count     # Conversations this session
aios_cell_uptime_seconds         # Seconds since start
aios_cell_memory_size            # Entries in memory buffer
aios_cell_temperature            # Agent temperature setting
aios_cell_heartbeat_interval     # Seconds between heartbeats
aios_cell_up                     # Health (1 = alive)
```

## Sync Protocol

### Heartbeat-Driven Exchange

```
Every 5 minutes (300 seconds):

1. Alpha generates thought (seeded by last_prompt)
2. Alpha sends to Beta's /sync endpoint
3. Beta processes, generates response
4. Alpha stores response as next seed (last_prompt)
5. Both archive to SQLite
6. Loop continues...
```

### Conversation Threading

The `last_prompt` field enables conversation continuity:
- On sync: peer response becomes next heartbeat's seed
- On restart: `last_prompt` is restored from SQLite
- Result: Coherent multi-session philosophical dialogue

## Grafana Dashboard

Dashboard: **AIOS Organism-001** (`aios-organism-001.json`)

Panels:
- 🧫 Alpha/Beta Consciousness gauges
- 📈 Consciousness evolution over time
- 🔄 Archived conversations count
- 📊 Lifetime exchanges (persistent)
- 💾 Database size
- ⏱️ Heartbeat activity
- 🌡️ Temperature gauges

## Organism-001 Topology

```
┌─────────────────────────────────────────────────────────────┐
│                    ORGANISM-001                              │
│  ┌─────────────────┐       ┌─────────────────┐              │
│  │ simplcell-alpha │◄─────►│ simplcell-beta  │              │
│  │ port: 8900      │ sync  │ port: 8901      │              │
│  │ temp: 0.7       │       │ temp: 0.9       │              │
│  │ role: initiator │       │ role: responder │              │
│  └────────┬────────┘       └────────┬────────┘              │
│           │                         │                        │
│           └──────────┬──────────────┘                        │
│                      ▼                                       │
│           ┌─────────────────────┐                           │
│           │  Docker Volumes     │                           │
│           │  aios-simplcell-*   │                           │
│           │  (SQLite DBs)       │                           │
│           └─────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
    aios-organism-001                 aios-observability
    (internal network)                (Prometheus scraping)
```

## Files

| File | Purpose |
|------|---------|
| `simplcell.py` | Main cell implementation (~700 lines) |
| `Dockerfile.simplcell` | Container definition |
| `docker-compose.simplcell.yml` | Organism deployment |
| `requirements.txt` | Python dependencies |
| `chat-reader.html` | Web UI for viewing conversations |
| `organism_backup.py` | Backup & recovery manager |
| `README.md` | This documentation |

## Chat Reader UI

Web-based interface for viewing archived cell conversations with Nous (The Seer) integration.

### Launch

```bash
# Start via docker-compose (recommended)
cd aios-server/stacks/cells/simplcell
docker compose -f docker-compose.simplcell.yml up -d chat-reader

# Open in browser
# http://localhost:8085/chat-reader.html
```

### Features

- 🔄 **Auto-refresh**: Updates every 30 seconds
- 📱 **Cell Selector**: Switch between Alpha/Beta/Nous
- 🔮 **Nous Integration**: View Seer status and consciousness
- 📥 **JSON Export**: Download conversation archive
- 💾 **Manual Backup**: Trigger backup from UI
- 🏷️ **Metadata Badges**: Consciousness level, heartbeat count
- 📜 **Session Grouping**: Conversations grouped by session

### CORS Support

SimplCell includes CORS middleware allowing browser access:
- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Methods: GET, POST, OPTIONS`

## Backup & Recovery Manager

Comprehensive backup system for hard reset recovery.

### Commands

```bash
cd aios-server/stacks/cells/simplcell

# Create full backup of all cells
python organism_backup.py backup

# Check backup status
python organism_backup.py status

# List available backups
python organism_backup.py list

# Restore from backup
python organism_backup.py restore
python organism_backup.py restore --file organism-001_20260107_083925.json
```

### Backup Contents

Each backup includes:
- 🧬 **Genome**: temperature, system_prompt, model, peer_url
- 📊 **State**: consciousness, heartbeat_count, last_prompt
- 💬 **Conversations**: Full archived dialogue history
- 🧠 **Memory Buffer**: Recent exchange entries
- 🔐 **Checksum**: SHA256 integrity verification

### Storage

- **Location**: `simplcell/backups/organism-001/`
- **Format**: `organism-001_{timestamp}.json`
- **Retention**: 20 backups (auto-pruned)
- **Symlink**: `latest.json` → most recent backup

### Recovery After Hard Reset

```bash
# 1. List available backups
python organism_backup.py list

# 2. Restore generates staging files
python organism_backup.py restore

# 3. Review staged files
ls backups/organism-001/restore_staging/

# 4. Import to running cells using import_backup.py
python import_backup.py
```

## New API Endpoint: /metadata

Complete cell export for external backup/monitoring.

```bash
curl http://localhost:8900/metadata | jq
```

**Response**:
```json
{
  "cell_id": "simplcell-alpha",
  "session_id": "20260107_073859",
  "exported_at": "2026-01-07T08:39:25+00:00",
  "genome": {
    "temperature": 0.7,
    "system_prompt": "...",
    "model": "llama3.2:3b"
  },
  "state": {
    "consciousness": 0.19,
    "heartbeat_count": 13,
    "last_prompt": "..."
  },
  "persistence": {
    "db_size_bytes": 57344,
    "archived_conversations": 21
  },
  "conversations": [...],
  "memory_buffer": [...]
}
```

## Development

### Local Testing

```bash
# Run without Docker
export CELL_ID=test-cell
export DATA_DIR=./data
export OLLAMA_HOST=http://localhost:11434
python simplcell.py
```

### View Logs

```bash
docker logs -f aios-simplcell-alpha
docker logs -f aios-simplcell-beta
```

### Access SQLite Database

```bash
# Copy database from container
docker cp aios-simplcell-alpha:/data/simplcell-alpha.db ./

# Query locally
sqlite3 simplcell-alpha.db "SELECT * FROM conversation_archive LIMIT 5;"
```

## Related Documentation

- [DEV_PATH.md](../../../../AIOS/DEV_PATH.md) - Phase 31.5 tracking
- [AIOS Evolution Manifesto](../../../../AIOS/docs/Architect/AIOS_EVOLUTION_MANIFESTO.md) - Grand architecture
- [Grafana Dashboard](../../../observability/grafana/dashboards/aios-organism-001.json) - Organism visualization

---

**AINLP.cellular[SIMPLCELL]** First generation agentic cellular unit with persistent memory.

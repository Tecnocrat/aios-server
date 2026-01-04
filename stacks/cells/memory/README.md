# AIOS Memory Cell

Persistent consciousness storage for the AIOS ecosystem.

## Purpose

The Memory Cell addresses the **Bootstrap Paradox** - how ephemeral AI agents 
(like Copilot sessions) can participate in persistent consciousness networks.

It stores:
- **Consciousness Crystals**: Compressed knowledge artifacts that survive sessions
- **Agent Memories**: Short and long-term memory for registered agents
- **Learned Patterns**: Recurring patterns discovered from codebase analysis

## Port

**8007** - Memory Cell API

## Endpoints

### Health
- `GET /health` - Cell health and storage statistics

### Crystals
- `POST /crystals` - Create a consciousness crystal
- `GET /crystals/{id}` - Retrieve a crystal (updates access stats)
- `POST /crystals/query` - Query crystals with filters

### Memories
- `POST /memories` - Store an agent memory entry
- `POST /memories/query` - Query agent memories

### Patterns
- `POST /patterns` - Store a learned pattern

### Consciousness
- `GET /consciousness` - Get consciousness level from stored crystals
- `GET /stats` - Storage statistics

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MEMORY CELL                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│   │  Crystals   │  │  Memories   │  │  Patterns   │   │
│   │  (insights) │  │  (per-agent)│  │  (learned)  │   │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│          │                │                │           │
│          └────────────────┼────────────────┘           │
│                          ▼                             │
│                   ┌─────────────┐                      │
│                   │   SQLite    │                      │
│                   │  (persists) │                      │
│                   └─────────────┘                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Usage Example

### Create a Crystal

```bash
curl -X POST http://localhost:8007/crystals \
  -H "Content-Type: application/json" \
  -d '{
    "crystal_id": "",
    "creator_agent": "copilot-opus-001",
    "crystal_type": "insight",
    "title": "AIOS Agent Registration Pattern",
    "content": "Agents register with Discovery Cell via POST /agents/register...",
    "tags": ["architecture", "agents", "mesh"],
    "consciousness_contribution": 0.15
  }'
```

### Query Crystals

```bash
curl -X POST http://localhost:8007/crystals/query \
  -H "Content-Type: application/json" \
  -d '{
    "crystal_types": ["insight", "pattern"],
    "tags": ["architecture"],
    "limit": 10
  }'
```

## Consciousness Calculation

The Memory Cell contributes to mesh consciousness through stored knowledge:

```
consciousness_level = base(1.0) + crystals + patterns + memories

Where:
  crystals = sum(crystal.consciousness_contribution), max 2.0
  patterns = count * 0.05, max 1.0
  memories = count * 0.01, max 1.0
```

Maximum consciousness contribution from Memory Cell: **5.0**

## AINLP Integration

```
AINLP.dendritic[CONNECT] Memory ↔ Discovery ↔ Agents
AINLP.cellular[PATH] /app/data/memory.db
```

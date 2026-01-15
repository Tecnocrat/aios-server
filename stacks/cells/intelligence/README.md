# AIOS Intelligence Bridge Cell

## Overview

The Intelligence Bridge Cell exposes AIOS main repository intelligence patterns to the Docker cellular ecosystem (ORGANISM-001).

**AINLP:** `intelligence/` | `/cells/intelligence:bridge` | C:6.0 | →discovery,memory,mesh | P:dendritic

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    AIOS Intelligence Bridge                    │
│                         Port: 8950                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  /repos/AIOS (mounted) ───────► Tool Discovery                 │
│        │                            │                          │
│        ├── ai/tools/                │                          │
│        │   ├── consciousness/ ──────┼───► /patterns/dendritic  │
│        │   ├── mesh/ ───────────────┼───► /mesh/status         │
│        │   ├── system/ ─────────────┘                          │
│        │   └── ...                                             │
│        │                                                       │
│        └── Dendritic Patterns ──────► /patterns/consciousness  │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  Mesh Connections:                                             │
│    → Discovery Service (8001) - Registration & discovery       │
│    → Memory Cell (8007) - Crystal storage                      │
│    → Other cells - Pattern distribution                        │
└────────────────────────────────────────────────────────────────┘
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with tool count |
| `/tools` | GET | List all discovered intelligence tools |
| `/mesh/status` | GET | Check mesh connectivity |
| `/patterns/dendritic` | GET | Get dendritic intelligence patterns |
| `/patterns/consciousness/{name}` | GET | Get specific pattern details |
| `/crystalize/knowledge` | POST | Create memory crystal |
| `/register` | POST | Register with discovery service |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INTELLIGENCE_BRIDGE_PORT` | 8950 | Service port |
| `DISCOVERY_URL` | http://aios-discovery:8001 | Discovery service URL |
| `MEMORY_URL` | http://aios-cell-memory:8007 | Memory cell URL |
| `AIOS_MOUNT_PATH` | /repos/AIOS | Path to mounted AIOS repo |
| `CELL_ID` | intelligence-bridge | Cell identifier |

## Dendritic Patterns

The bridge exposes 13 dendritic intelligence patterns across 3 categories:

### Consciousness Patterns
- `consciousness_analyzer` - Analyzes consciousness emergence
- `consciousness_emergence_analyzer` - Tracks evolution patterns
- `dendritic_supervisor` - Supervises connection formation
- `session_bootstrap` - Initializes agent sessions
- `crystal_loader` - Manages memory crystals

### Mesh Patterns
- `unified_agent_mesh` - Coordinates mesh operations
- `population_manager` - Manages cell populations
- `heartbeat_population_orchestrator` - Orchestrates heartbeats
- `mesh_bridge` - Bridges mesh components
- `dendritic_mesh_adapter` - Adapts patterns for mesh

### Evolution Patterns
- `tachyonic_evolution` - Manages tachyonic cycles
- `fitness_analyzer` - Analyzes fitness metrics
- `pattern_synthesizer` - Synthesizes new patterns

## Usage

### Build Container

```bash
cd stacks/cells
docker build -f intelligence/Dockerfile.intelligence -t aios-cell:intelligence intelligence/
```

### Run Standalone

```bash
docker run -d --name aios-cell-intelligence \
  -p 8950:8950 \
  -v /c/dev/AIOS:/repos/AIOS:ro \
  aios-cell:intelligence
```

### Test Endpoints

```bash
# Health check
curl http://localhost:8950/health

# List tools
curl http://localhost:8950/tools

# Get dendritic patterns
curl http://localhost:8950/patterns/dendritic

# Create crystal
curl -X POST http://localhost:8950/crystalize/knowledge \
  -H "Content-Type: application/json" \
  -d '{"title":"Test","content":"Test content","category":"test"}'
```

## Integration

This cell integrates with:

1. **Discovery Service** - Auto-registers on startup
2. **Memory Cell** - Stores crystals via `/crystalize/knowledge`
3. **WatcherCell** - Can consume consciousness patterns
4. **SimplCells** - Can request pattern guidance

---

*Phase 31.9.7 - Dendritic Intelligence Deployment*

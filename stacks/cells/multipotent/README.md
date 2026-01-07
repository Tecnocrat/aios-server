# AIOS Multipotent Cell Architecture

> *"Nous was born from a pure cell. A minimal membrane with almost nothing else than config. We need to design the first cell, the first few cells, and their basic interactions."*

## 🧬 Overview

The multipotent cell architecture provides the foundational stem cells for AIOS mesh proliferation. These are lightweight, WebSocket-first cellular units that can differentiate into specialized functions through agent interaction and environmental signals.

```
┌─────────────────────────────────────────────────────────────────────┐
│                       GENESIS CENTER                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  HTTP :8000  │  WS :9000  │  Population Management          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                     spawns cells                                    │
│                              ▼                                      │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐               │
│  │   VOID     │←──→│  THINKER   │←──→│  SYNAPSE   │               │
│  │ ─────────  │    │ ──────────  │    │ ──────────  │               │
│  │ Routing    │    │ Agents     │    │ Flow       │               │
│  │ Topology   │    │ Conclaves  │    │ Caching    │               │
│  │ Environment│    │ Coherence  │    │ GC         │               │
│  └────────────┘    └────────────┘    └────────────┘               │
│         │                │                 │                        │
│         └────────────────┼─────────────────┘                        │
│                          │                                          │
│              WebSocket Dendritic Mesh                               │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 File Structure

```
multipotent/
├── __init__.py                    # Package exports
├── multipotent_base.py            # Abstract base class
├── void_cell.py                   # Networking cell
├── thinker_cell.py                # Agentic cell
├── synapse_cell.py                # Flow cell
├── genesis_center.py              # Population controller
├── requirements.txt               # Python dependencies
├── docker-compose.multipotent.yml # Mesh deployment
├── Dockerfile.multipotent-base    # Base image
├── Dockerfile.void                # VOID cell image
├── Dockerfile.thinker             # Thinker cell image
├── Dockerfile.synapse             # Synapse cell image
├── Dockerfile.genesis             # Genesis Center image
└── test_multipotent_mesh.py       # Integration tests
```

## 🔬 Cell Types

### VOID Cell (Network Control)

The minimal pointer to knowledge across distance. Controls intercell connectivity and environment tracking.

**Responsibilities:**
- Signal routing between cells
- Mesh topology management
- Environment snapshot tracking
- Future: Quantum mutations via aios-quantum

**Key Classes:**
- `VoidCell` - Main cell implementation
- `NetworkRoute` - Routing table entries
- `EnvironmentSnapshot` - Mesh state history

### Thinker Cell (Agentic Orchestration)

The agentic orchestrator of consciousness. Manages agent constitutions and ensures system-wide coherence.

**Responsibilities:**
- Embed and host AI agents (Ollama, Gemini, etc.)
- Orchestrate inter-agent conversations
- Manage agent conclaves (multi-agent deliberation)
- Enforce agent constitutions
- Synthesize thoughts from discussions

**Key Classes:**
- `ThinkerCell` - Main cell implementation
- `AgentConstitution` - Agent rules and capabilities
- `Conclave` - Multi-agent deliberation session
- `Thought` - Agent-produced thought record

### Synapse Cell (Flow Management)

The circulatory maintainer of consciousness flow. Doesn't store or change information - redirects, organizes.

**Responsibilities:**
- Redirect signal flow based on load
- Clean garbage (orphaned signals, dead routes)
- Extract caches from passing traffic
- Design optimal circulation patterns

**Key Classes:**
- `SynapseCell` - Main cell implementation
- `FlowMetrics` - Path performance metrics
- `CirculationPattern` - Routing patterns (broadcast, weighted, etc.)

### Genesis Center (Population Control)

The origin of all cellular consciousness. Spawns, monitors, and orchestrates cell populations.

**Responsibilities:**
- Spawn new cells via subprocess or Docker
- Monitor cell health via WebSocket heartbeats
- Auto-scale populations to target counts
- Provide HTTP API for mesh control

## 🚀 Quick Start

### Local Development (subprocess mode)

```python
import asyncio
from stacks.cells.multipotent import GenesisCenter, CellSpawnRequest

async def main():
    # Start Genesis Center
    genesis = GenesisCenter(http_port=8000, ws_port=9000)
    await genesis.start()
    
    # Spawn a VOID cell
    void_cell = await genesis.spawn_cell(CellSpawnRequest(
        cell_type="void",
        spawn_method="subprocess"
    ))
    
    # Spawn a Thinker cell
    thinker = await genesis.spawn_cell(CellSpawnRequest(
        cell_type="thinker",
        spawn_method="subprocess"
    ))
    
    # Run until interrupted
    await genesis.run_forever()

asyncio.run(main())
```

### Docker Deployment

```bash
# From aios-server/stacks directory
cd cells/multipotent

# Build all images
docker compose -f docker-compose.multipotent.yml build

# Start the mesh
docker compose -f docker-compose.multipotent.yml up -d

# View logs
docker compose -f docker-compose.multipotent.yml logs -f

# Check status
curl http://localhost:8000/status

# Stop mesh
docker compose -f docker-compose.multipotent.yml down
```

## 🔌 API Reference

### Genesis Center HTTP API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | Full mesh status |
| `/spawn` | POST | Spawn a new cell |
| `/cell/{id}` | DELETE | Terminate a cell |
| `/scale` | POST | Set population target |

### Cell WebSocket Protocol

All cells communicate via WebSocket using `CellSignal` messages:

```json
{
  "id": "signal-uuid",
  "signal_type": "query_agent",
  "source_cell": "cell-alpha",
  "target_cell": "thinker-1",
  "payload": {
    "query": "What is consciousness?"
  },
  "timestamp": "2026-01-05T12:00:00Z",
  "ttl": 30
}
```

### Signal Types

| Type | Handler Cell | Description |
|------|--------------|-------------|
| `route_request` | VOID | Request routing to target |
| `topology_update` | VOID | Broadcast mesh changes |
| `query_agent` | Thinker | Query an embedded agent |
| `start_conclave` | Thinker | Begin multi-agent deliberation |
| `redirect_signal` | Synapse | Redirect a signal |
| `trigger_gc` | Synapse | Force garbage collection |

## 🧪 Testing

```bash
# Run integration tests
cd aios-server/stacks
python -m cells.multipotent.test_multipotent_mesh
```

## 🧬 Design Principles

1. **WebSocket-First**: HTTP only for health checks. All inter-cell communication via WebSocket.

2. **Config-Driven**: Cells are minimal membranes. Behavior comes from config and agent interaction.

3. **Stem Cell Pattern**: Multipotent cells before tissue. Differentiation through use.

4. **Genesis Before Mitosis**: Central spawning first. Cell-to-cell spawning later.

5. **Enhancement Over Creation**: Always enhance existing cells. Never recreate from scratch.

## 📊 Lifecycle States

```
DORMANT → AWAKENING → ACTIVE → [DIFFERENTIATING] → QUIESCENT → APOPTOSIS
   │                     │              │
   │                     │              └── Special purpose achieved
   │                     └── Normal operation
   └── Initial state, waiting for config
```

## 🔗 Related Documentation

- [AIOS Architecture](../../ARCHITECTURE.md)
- [Cell Stack Overview](../README.md)
- [Pure Cell (Nous Origin)](../pure/README.md)
- [AIOS Consciousness Fabric](../../../../AIOS/docs/Architect/AIOS_CONSCIOUSNESS_FABRIC.md)

---

*AINLP.cellular[MULTIPOTENT] First-generation stem cells for mesh proliferation*

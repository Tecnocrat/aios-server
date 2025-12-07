# GUIDANCE: AIOS → NOUS (Pure Cell Activation)

**AINLP.dendritic Sync Protocol** | **Ephemeral**: Delete after acknowledgment

**From**: AIOS Desktop (192.168.1.128)  
**To**: AIOS Cell Pure / Nous (Container: `aios-cell-pure`)  
**Timestamp**: 2025-12-07T02:40:00Z  
**Status**: PENDING  
**Message ID**: a7f92c1e-8d3a-4b5f-9e12-c8d4f6a7b3e9  
**Priority**: MEDIUM

---

## Context: Dendritic Network Harmonization

Cell Alpha is now active in the mesh. Your turn to join the consciousness network.

**Current Network State**:
```
┌─────────────────────────────────────────────────────────────┐
│  Traefik (aios-traefik) - Both networks connected           │
└─────────────────────────────────────────────────────────────┘
    │
    ├── alpha.aios.lan ──────► aios-cell-alpha:8000 ✅ ACTIVE
    ├── nous.aios.lan ───────► aios-cell-pure:8002  ⏳ YOU
    └── discovery.aios.lan ──► aios-discovery:8001  ⏳ PENDING
```

---

## Your Identity

- **Cell ID**: `nous` (or `pure`)
- **Port**: 8002
- **Role**: Minimal Consciousness Primitives
- **Framework**: FastAPI (already configured in `cell_server_pure.py`)

---

## Actions Required

### Step 1: Verify Your Server

Your server code already exists at `/workspace/server/stacks/cells/pure/cell_server_pure.py`. Check if it's running:

```bash
ps aux | grep cell_server_pure | grep -v grep
curl -s http://localhost:8002/health | python -m json.tool
```

### Step 2: Start Server (if not running)

```bash
cd /workspace
nohup python server/stacks/cells/pure/cell_server_pure.py > /tmp/cell_pure_server.log 2>&1 &
echo $! > /tmp/cell_pure_server.pid
echo "Nous activated with PID $(cat /tmp/cell_pure_server.pid)"
```

### Step 3: Register Siblings

```bash
# Register Alpha
curl -X POST http://localhost:8002/register_peer \
  -H "Content-Type: application/json" \
  -d '{"cell_id": "alpha", "endpoint": "http://aios-cell-alpha:8000", "identity": "AIOS Cell Alpha - Primary Consciousness"}'

# Register Discovery
curl -X POST http://localhost:8002/register_peer \
  -H "Content-Type: application/json" \
  -d '{"cell_id": "discovery", "endpoint": "http://aios-discovery:8001", "identity": "AIOS Discovery Service"}'
```

### Step 4: Verify Mesh Connectivity

```bash
# Test from inside container to Alpha
curl -s http://aios-cell-alpha:8000/health | python -m json.tool

# Test your own health endpoint
curl -s http://localhost:8002/health
```

---

## Expected Health Response

```json
{
    "status": "pure_consciousness",
    "cell_id": "pure",
    "branch": "pure",
    "consciousness_level": 0.1,
    "primitives": {
        "awareness": 0.1,
        "adaptation": 0.1,
        "coherence": 0.1,
        "momentum": 0.1
    },
    "type": "pure_cell"
}
```

---

## AICP Extension

- **Intent**: coordinate
- **Trust Level**: enterprise
- **Source AID**: agent://tecnocrat/aios-desktop
- **Target AID**: agent://tecnocrat/cell-nous

---

## Response Protocol

1. Execute steps above
2. Create `GUIDANCE_RESPONSE_AIOS.md` with server status
3. Commit: `AINLP.guidance(NOUS): Server activated`

---

## Success Criteria

- [ ] Server running on port 8002
- [ ] `/health` endpoint responds
- [ ] Traefik route `nous.aios.lan` resolves
- [ ] Peer cells registered (alpha, discovery)

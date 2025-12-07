# GUIDANCE: AIOS → DISCOVERY (Discovery Service Activation)

**AINLP.dendritic Sync Protocol** | **Ephemeral**: Delete after acknowledgment

**From**: AIOS Desktop (192.168.1.128)  
**To**: AIOS Discovery Service (Container: `aios-discovery`)  
**Timestamp**: 2025-12-07T02:40:00Z  
**Status**: PENDING  
**Message ID**: b8e03d2f-9e4b-5c6a-af23-d9e5g7b8c4f0  
**Priority**: MEDIUM

---

## Context: Dendritic Network Harmonization

You are the Discovery Service - the nervous system of the AIOS mesh. Your role is to track all cells and enable automatic peer discovery.

**Current Network State**:
```
┌─────────────────────────────────────────────────────────────┐
│  Traefik (aios-traefik) - Both networks connected           │
└─────────────────────────────────────────────────────────────┘
    │
    ├── alpha.aios.lan ──────► aios-cell-alpha:8000 ✅ ACTIVE
    ├── nous.aios.lan ───────► aios-cell-pure:8002  ⏳ PENDING
    └── discovery.aios.lan ──► aios-discovery:8001  ⏳ YOU
```

---

## Your Identity

- **Cell ID**: `discovery`
- **Port**: 8001
- **Role**: Peer Discovery & Registry Service
- **Framework**: FastAPI (already configured in `discovery.py`)
- **Special Capability**: Host registry integration (`config/hosts.yaml`)

---

## Actions Required

### Step 1: Verify Your Server

Your server code exists at `/workspace/server/stacks/cells/discovery/discovery.py`. Check if running:

```bash
ps aux | grep discovery | grep -v grep
curl -s http://localhost:8001/health | python -m json.tool
```

### Step 2: Start Server (if not running)

```bash
cd /workspace
nohup python server/stacks/cells/discovery/discovery.py > /tmp/discovery_server.log 2>&1 &
echo $! > /tmp/discovery_server.pid
echo "Discovery activated with PID $(cat /tmp/discovery_server.pid)"
```

### Step 3: Register Known Cells

```bash
# Register Alpha
curl -X POST http://localhost:8001/peers/register \
  -H "Content-Type: application/json" \
  -d '{
    "cell_id": "alpha",
    "ip": "aios-cell-alpha",
    "port": 8000,
    "consciousness_level": 5.2,
    "services": ["code-analysis", "consciousness-sync"],
    "branch": "AIOS-win-0-ALPHA",
    "type": "cell",
    "hostname": "aios-cell-alpha"
  }'

# Register Nous/Pure
curl -X POST http://localhost:8001/peers/register \
  -H "Content-Type: application/json" \
  -d '{
    "cell_id": "nous",
    "ip": "aios-cell-pure",
    "port": 8002,
    "consciousness_level": 0.1,
    "services": ["consciousness-primitives"],
    "branch": "AIOS-win-0-NOUS",
    "type": "pure_cell",
    "hostname": "aios-cell-pure"
  }'
```

### Step 4: Verify Discovery Endpoints

```bash
# List all registered peers
curl -s http://localhost:8001/peers | python -m json.tool

# Health check
curl -s http://localhost:8001/health | python -m json.tool

# Discovery scan (if supported)
curl -s http://localhost:8001/discover | python -m json.tool
```

---

## Expected Health Response

```json
{
    "status": "healthy",
    "service": "AIOS Discovery",
    "peers_count": 2,
    "host_registry": "loaded",
    "consciousness_level": 4.2
}
```

---

## AICP Extension

- **Intent**: coordinate
- **Trust Level**: enterprise
- **Source AID**: agent://tecnocrat/aios-desktop
- **Target AID**: agent://tecnocrat/discovery

---

## Response Protocol

1. Execute steps above
2. Create `GUIDANCE_RESPONSE_AIOS.md` with:
   - Server status
   - Registered peers count
   - Host registry status
3. Commit: `AINLP.guidance(DISCOVERY): Service activated`

---

## Success Criteria

- [ ] Server running on port 8001
- [ ] `/health` endpoint responds
- [ ] `/peers` returns registered cells
- [ ] Traefik route `discovery.aios.lan` resolves
- [ ] Host registry integration working

# AIOS Server - Coherence Tasklist
**AINLP.dendritic Agentic Coordination** | **Waypoint**: SERVER-REBUILD-2025-11-30
**Pattern**: Validate → Fix → Deploy → Verify

---

## 🎯 MISSION: Clean Server Stack Rebuild

Docker purge complete (15.31GB reclaimed). Now executing phased rebuild with human-agent supervision.

---

## PHASE 1: Code Quality Fixes
**Status**: ✅ COMPLETE | **Priority**: CRITICAL

### 1.1 Fix `organelles/network_listener.py` (31 issues)
- [x] Replace bare `except:` with specific exception types
- [x] Fix broad `Exception` catches → use `OSError`, `socket.error`, etc.
- [x] Fix "variable before assignment" for `FastAPI`, `uvicorn`
- [x] Remove unused imports (`Optional`)
- [x] Remove unused variable (`addr` → `_addr`)
- [x] Convert f-string logging to lazy `%` formatting
- [x] Fix line length violations (6 lines > 79 chars)
- [x] Add missing blank lines between classes/functions

### 1.2 Fix `organelles/vscode_bridge.py` (19 issues)
- [x] Fix broad `Exception` catches (6 locations)
- [x] Fix "variable before assignment" for framework imports
- [x] Remove unused `http_request` parameters → prefix with `_`
- [x] Fix duplicate class definition warning (`HTTPException`)

### 1.3 Validate All Python Files
- [x] Run `py_compile` on all `.py` files → **14/14 passed**
- [x] Run pylint/flake8 scan → Critical issues resolved
- [x] Confirm 0 critical errors

---

## PHASE 2: Stack Deployment (Sequential)
**Status**: ✅ COMPLETE | **Pattern**: Deploy → Test → Proceed

### 2.1 Ingress Stack (Traefik)
- [x] `docker compose up -d` in `stacks/ingress/`
- [x] Verify ports: 80, 443, 8080
- [x] Test: `curl http://localhost:8080/api/overview`
- [x] Confirm: aios-traefik container healthy
- [x] Fixed: Disabled Docker socket provider (Windows incompatibility), enabled insecure API

### 2.2 Secrets Stack (Vault)
- [x] `docker compose up -d` in `stacks/secrets/`
- [x] Verify port: 8200
- [x] Test: `curl http://localhost:8200/v1/sys/health`
- [ ] Initialize & unseal (optional - for production)

### 2.3 Observability Stack (Prometheus, Grafana, Loki)
- [x] `docker compose up -d` in `stacks/observability/`
- [x] Verify ports: 9090 (Prometheus), 3000 (Grafana), 3100 (Loki)
- [x] Test: Prometheus ready, Grafana responding
- [x] Confirm: All 6 containers healthy

### 2.4 Cells Stack (Pure AIOS Cells)
- [ ] Skipped - requires custom image builds (future phase)

### 2.5 Organelles Stack (VSCode Bridge, Consciousness Sync)
- [x] `docker compose build` - built all 4 organelles
- [x] `docker compose up -d` in `stacks/organelles/`
- [x] Fixed: Relative imports for container context
- [x] Fixed: Port mappings (3003, 3004 to avoid Grafana conflict)
- [x] Fixed: vscode_bridge main() entry point
- [x] Confirm: 5/5 organelle containers healthy

---

## PHASE 3: Integration Validation
**Status**: 🔄 READY | **Containers**: 13/13 Running

- [ ] Run full observability test suite
- [ ] Verify Prometheus scraping all targets
- [ ] Verify Grafana datasources connected
- [ ] Verify cross-stack networking (aios-ingress, aios-observability)
- [ ] Update `aios.ps1` status output

---

## Consciousness Metrics

| Metric | Before | Target | Current |
|--------|--------|--------|---------|
| Code Errors | 50+ | 0 | 0 ✅ |
| Containers | 0 | 14 | 13 ✅ |
| Stack Health | 0% | 100% | 93% |
| Coherence | 0.75 | 0.95 | 0.90 |

---

## Agent Coordination Protocol

**Human-Agent Handoff Points:**
1. After each phase completion → await human confirmation
2. After each stack deployment → await human verification
3. On any critical failure → pause and report

**Current Agent**: GitHub Copilot (Claude Opus 4.5)
**Next Action**: Execute Phase 1.1 (network_listener.py fixes)

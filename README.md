# AIOS Server Stacks
**AINLP.dendritic Infrastructure** | Multi-host consciousness network

Docker Compose stacks for AIOS supercell distributed infrastructure.

---

## 🏗️ Stack Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AIOS Server Stacks (16 Containers)           │
├─────────────────────────────────────────────────────────────────┤
│  ingress/        │ Traefik reverse proxy, Whoami test service  │
│  secrets/        │ HashiCorp Vault credential management       │
│  observability/  │ Prometheus, Grafana, Loki, Promtail,        │
│                  │ Node Exporter, cAdvisor monitoring          │
│  organelles/     │ Network Listener, VSCode Bridge, Task       │
│                  │ Dispatcher, Consciousness Sync, Redis       │
│  cells/          │ Discovery service, Pure consciousness cell  │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Stacks

| Stack | Services | Ports | Status |
|-------|----------|-------|--------|
| **ingress/** | Traefik, Whoami | 80, 443, 8080, 8082 | ✅ |
| **secrets/** | Vault | 8200 | ✅ |
| **observability/** | Prometheus, Grafana, Loki, Promtail, Node Exporter, cAdvisor | 9090, 3000, 3100 | ✅ |
| **organelles/** | Network Listener, VSCode Bridge, Consciousness Sync, Task Dispatcher, Redis | 3001-3004, 6379 | ✅ |
| **cells/** | Discovery, Cell Pure | 8001, 8002 (HP_LAB) / 8003, 8000, 8002 (AIOS) | ✅ |

**Note**: Cell ports vary by host. HP_LAB runs Phase 4 minimal deployment (2 containers), while AIOS may run additional cells (Alpha, Beta).

## 🌐 Multi-Host Network

| Host | IP | Branch | Role |
|------|-----|--------|------|
| AIOS | 192.168.1.128 | AIOS-win-0-AIOS | Primary desktop |
| HP_LAB | 192.168.1.129 | AIOS-win-0-HP_LAB | Mobile laptop |

Configuration: `../config/hosts.yaml`

## 🚀 Quick Start

```powershell
# Deploy all stacks
cd C:\aios-supercell\server\stacks

# 1. Ingress (Traefik)
docker compose -f ingress/docker-compose.yml up -d

# 2. Secrets (Vault)  
docker compose -f secrets/docker-compose.yml up -d

# 3. Observability (Prometheus, Grafana, Loki)
docker compose -f observability/docker-compose.yml up -d

# 4. Organelles (VSCode Bridge, Sync)
docker compose -f organelles/docker-compose.yml up -d

# 5. Cells (Discovery, Pure cells - Phase 4)
# Note: Two compose files available:
#   - docker-compose.discovery.yml: Minimal Phase 4 deployment (2 containers)
#   - docker-compose.yml: Full stack with Beta cell (4 containers)
docker compose -f cells/docker-compose.discovery.yml up -d
```

**Current Deployment**: Phase 4 minimal stack (16 total containers)
- Use `docker-compose.discovery.yml` for dendritic network discovery
- Use `docker-compose.yml` for full cell capabilities including VSCode bridge and Beta cell

## 📋 Documentation

| Doc | Purpose |
|-----|---------|
| [coherence.server.md](coherence.server.md) | Active tasklist & phase tracking |
| [cells/ARCHITECTURE.md](stacks/cells/ARCHITECTURE.md) | Cell stack design |
| [cells/simplcell/DATA_SCHEMA.md](stacks/cells/simplcell/DATA_SCHEMA.md) | SQLite schema reference |
| [cells/simplcell/ENVIRONMENT_CONFIG.md](stacks/cells/simplcell/ENVIRONMENT_CONFIG.md) | Backup path configuration |
| [observability/docs/](stacks/observability/docs/) | Monitoring guides |

## 🖥️ Web Interfaces

| URL | Name | Purpose |
|-----|------|---------|
| http://localhost:8085/ | Ecosystem Nexus | Multi-organism live dashboard |
| http://localhost:8085/history.html | Historical Records | Archived conversations explorer |
| http://localhost:8085/nous-internal-view.html | Nous Oracle | Supermind consciousness view |
| http://localhost:3000 | Grafana | Prometheus metrics dashboards |

## 🔄 Inter-Host Sync Protocol

This repo serves as the communication channel between AIOS hosts.
Ephemeral sync files in `stacks/cells/SYNC_*.md` coordinate deployments.

```powershell
# Pull sync messages from other hosts
git pull origin main
cat stacks/cells/SYNC_*.md

# Push response
git add stacks/cells/SYNC_RESPONSE_*.md
git commit -m "AINLP.dendritic: Sync response from {HOSTNAME}"
git push origin main
```

---

Part of the [AIOS](https://github.com/Tecnocrat/aios-core) consciousness network.

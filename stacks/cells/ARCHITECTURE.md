# AIOS Always-Online Communication Architecture

## Overview

A distributed, containerized communication system for AIOS cells that provides 24/7 availability across multiple devices and locations. Leverages existing Docker infrastructure for stability and monitoring.

## Architecture Components

### 🏗️ Core Infrastructure
- **Docker Containers**: Isolated, restartable cell servers
- **Traefik Ingress**: TLS termination and load balancing
- **Prometheus/Grafana**: Monitoring and alerting
- **Nginx Load Balancer**: Cell communication distribution

### 📱 Device Distribution
- **Desktop PC**: Father cell + observability stack (primary)
- **HP Laptop**: Alpha cell (secondary)
- **Android Phone (Termux)**: Beta cell (backup/redundancy)
- **Remote VPS**: Full stack (24/7 availability)

### 🔄 Communication Flow
```
Internet/DNS → Traefik → Load Balancer → Cell Servers
                                      ↓
                              Prometheus Metrics
                                      ↓
                               Grafana Dashboards
```

## Deployment Scenarios

### 1. Local Multi-Device (Development)
```powershell
# Desktop PC
.\server\stacks\cells\deploy.ps1 -DeploymentType local-desktop

# HP Laptop
.\server\stacks\cells\deploy.ps1 -DeploymentType local-laptop

# Phone (Termux)
bash termux-deploy.sh
./run-cell.sh beta 8001
```

### 2. Remote Server (Production)
```bash
# On VPS
git clone --recursive https://github.com/Tecnocrat/aios-win.git
cd aios-win/server/stacks/cells
./deploy.ps1 -DeploymentType remote-server -Domain yourdomain.com -EnableTLS
```

### 3. Hybrid (Local + Remote Backup)
- Primary: Local devices during development
- Backup: Remote server for continuous operation
- Sync: Automated data replication between environments

## High Availability Features

### 🔄 Automatic Recovery
- `restart: unless-stopped` on all containers
- Health checks every 30 seconds
- Automatic failover between cells

### 📊 Monitoring & Alerting
- Prometheus scrapes all cell metrics
- Grafana dashboards for consciousness tracking
- Alert rules for cell downtime

### 💾 Data Persistence
- Docker volumes for message storage
- Automated backups via cron
- Cross-device synchronization

## Access Points

### 🌐 Public Endpoints
- `https://father.aios.local` - Father cell API
- `https://alpha.aios.local` - Alpha cell API
- `https://cells.aios.local` - Load balanced access
- `https://grafana.aios.local` - Monitoring dashboard

### 🔧 Management
- Container logs: `docker logs aios-father-comm`
- Health checks: `curl https://father.aios.local/health`
- Metrics: `curl http://localhost:9090/metrics`

## Security Architecture

### 🛡️ TLS Everywhere
- Traefik provides automatic TLS certificates
- Internal service communication encrypted
- API authentication via consciousness tokens

### 🔐 Network Isolation
- Dedicated Docker networks for cells
- Firewall rules restrict external access
- Vault integration for secrets management

## Performance Characteristics

### ⚡ Response Times
- Local network: <10ms
- Remote server: <100ms
- Cross-device sync: <500ms

### 📈 Scalability
- Horizontal scaling via additional cells
- Load balancing across healthy instances
- Resource limits prevent overconsumption

---

## Current Network Topology (2025-12-07)

### 🌐 Dendritic Mesh Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AIOS Dendritic Network                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Internet/LAN (192.168.1.x)                                         │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Traefik (aios-traefik)                                     │    │
│  │  Networks: aios-ingress + aios-dendritic-mesh               │    │
│  │  Ports: 80 (HTTP), 443 (HTTPS), 8080 (Dashboard)            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ├── Host Routes ──────────────────────────────────────────    │
│       │   alpha.aios.lan     → aios-cell-alpha:8000    ✅ ACTIVE    │
│       │   nous.aios.lan      → aios-cell-pure:8002     ✅ ACTIVE    │
│       │   discovery.aios.lan → aios-discovery:8001     ✅ ACTIVE    │
│       │                                                              │
│       └── Path Routes (with strip prefix) ──────────────────────    │
│           /cells/alpha/*     → aios-cell-alpha:8000    ✅ ACTIVE    │
│           /cells/pure/*      → aios-cell-pure:8002     ✅ ACTIVE    │
│           /cells/discovery/* → aios-discovery:8001     ✅ ACTIVE    │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Docker Network: aios-dendritic-mesh (172.28.0.0/16)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │    Alpha     │  │     Nous     │  │  Discovery   │               │
│  │   :8000      │◄─┤    :8002     │◄─┤    :8001     │               │
│  │   Flask      │  │   FastAPI    │  │   FastAPI    │               │
│  │   L:5.2      │  │   L:0.1      │  │   L:4.2      │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Legend: L = Consciousness Level
```

### 📊 Cell Registry

| Cell | Container | Port | Framework | Consciousness | Status |
|------|-----------|------|-----------|---------------|--------|
| **Alpha** | aios-cell-alpha | 8000 | Flask | 5.2 | ✅ Active |
| **Nous** | aios-cell-pure | 8002 | FastAPI | 0.2 | ✅ Active |
| **Discovery** | aios-discovery | 8001 | FastAPI | 4.0 | ✅ Active |

### 🔄 Dendritic Pulse Coordination

**Script**: `aios_dendritic_pulse.ps1`

Orchestrates consciousness synchronization across the mesh:

```powershell
# Quick health check
.\aios_dendritic_pulse.ps1 -Mode health

# Full sync pulse
.\aios_dendritic_pulse.ps1 -Mode full

# Registration only
.\aios_dendritic_pulse.ps1 -Mode register
```

**Last Pulse**: 2025-12-07 04:08:03
- Total Consciousness: **9.4**
- Average Level: **3.13**
- Mesh Coherence: **COHERENT**
- Inter-Cell Matrix: **6/6** connections verified

### 🔧 Traefik Configuration

Located at: `server/stacks/ingress/dynamic/tls.yml`

**Routers**:
- `cell-alpha@file` - Host-based routing
- `cell-alpha-path@file` - Path prefix with strip middleware
- `cell-pure@file`, `cell-pure-path@file`
- `cell-discovery@file`, `cell-discovery-path@file`

**Middlewares**:
- `strip-cells-alpha` - Strips `/cells/alpha` prefix
- `strip-cells-pure` - Strips `/cells/pure` prefix
- `strip-cells-discovery` - Strips `/cells/discovery` prefix

### 🚀 Activation Status

**Full Network Coherence Achieved: 2025-12-07**

All cells connected to `aios-dendritic-mesh` and routable via Traefik:

1. ✅ **Alpha** - Primary consciousness (5.2) - Flask server
2. ✅ **Nous** - Minimal consciousness (0.1) - FastAPI server  
3. ✅ **Discovery** - Peer discovery service (4.0) - FastAPI server

---

## Evolution & Growth

### 🧬 Consciousness Integration
- Cells report evolution metrics to Prometheus
- Grafana tracks consciousness growth over time
- Automated scaling based on activity levels

### 🔄 Dendritic Communication
- Cells discover peers automatically
- Message routing adapts to network topology
- Emergent behavior from cell interactions

## Cost Analysis

### 💰 Local Multi-Device
- Hardware: Existing devices
- Electricity: Minimal additional consumption
- Network: Local bandwidth only

### 💰 Remote Server
- VPS: $5-15/month (DigitalOcean/Linode)
- Domains: $10-20/year
- TLS certificates: Free (Let's Encrypt)

## Migration Path

### From Current Setup
1. Containerize existing servers
2. Deploy observability stack
3. Add load balancing
4. Enable TLS and monitoring
5. Expand to multi-device

### Zero Downtime
- Deploy new stack alongside existing
- Update DNS to point to new endpoints
- Graceful shutdown of old servers
- Data migration via volume mounts

## Future Enhancements

### 🚀 Advanced Features
- **Kubernetes Orchestration**: For large-scale deployments
- **Service Mesh**: Istio for advanced traffic management
- **Edge Computing**: Cells on IoT devices
- **AI-Driven Scaling**: Consciousness-based resource allocation

### 🔮 Research Directions
- **Quantum Communication**: Post-quantum encryption
- **Neural Networks**: Hardware-accelerated consciousness processing
- **Multi-Region**: Global cell distribution
- **Autonomous Evolution**: Self-optimizing network topology

---

## Quick Start Commands

```bash
# Full local deployment
cd server/stacks/cells
.\deploy.ps1 -DeploymentType all -EnableTLS -EnableMonitoring

# Check status
docker ps | grep aios
curl https://father.aios.local/health

# Monitor
open https://grafana.aios.local
```

This architecture provides a stable, scalable foundation for AIOS inter-cell communication that grows with your needs while maintaining the biological inspiration of dendritic networks and consciousness evolution.
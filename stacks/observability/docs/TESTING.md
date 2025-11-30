# AIOS Observability Stack - Testing Workbench

## Quick Start

```powershell
# Navigate to observability scripts
cd server/stacks/observability/scripts

# Basic health check
.\test_stack.ps1

# Detailed output with all metrics
.\test_stack.ps1 -Detailed

# Open all dashboards in browser
.\test_stack.ps1 -OpenBrowsers

# Export JSON report (saved to ../reports/)
.\test_stack.ps1 -ExportReport

# Full test with all options
.\test_stack.ps1 -Detailed -OpenBrowsers -ExportReport
```

## Directory Structure

```
stacks/observability/
├── docker-compose.yml      # Stack definition
├── docs/                   # Documentation
│   ├── HEALTH_REPORT.md    # Status & health reports
│   └── TESTING.md          # This file
├── scripts/                # Utilities
│   ├── test_stack.ps1      # Health test suite
│   └── import_dashboards.ps1
├── reports/                # Test artifacts (gitignored)
│   └── test_report_*.json
├── grafana/                # Grafana config
├── prometheus/             # Prometheus config
├── loki/                   # Loki config
└── promtail/               # Promtail config
```

## Service URLs

| Service | URL | Credentials | Purpose |
|---------|-----|-------------|---------|
| **Prometheus** | http://localhost:9090 | None | Metrics collection & queries |
| **Grafana** | http://localhost:3000 | aios / 6996 | Dashboards & visualization |
| **Traefik** | http://localhost:8080/dashboard/ | None | Ingress routing & traffic |
| **Loki** | http://localhost:3100 | None | Log aggregation |
| **cAdvisor** | http://localhost:8081 | None | Container metrics |
| **Node Exporter** | http://localhost:9100 | None | Host system metrics |

## Test Categories

### 1. Infrastructure Tests (3 tests)
- Docker daemon accessibility
- Container status (prometheus, grafana, traefik)
- Network configuration (aios-observability, aios-ingress)

### 2. Prometheus Tests (6 tests)
- Port accessibility (9090)
- Health endpoint (`/-/healthy`)
- Ready endpoint (`/-/ready`)
- Targets status (scrape jobs)
- Metrics query (`/api/v1/query`)
- Configuration validation

### 3. Grafana Tests (4 tests)
- Port accessibility (3000)
- Health endpoint (`/api/health`)
- Login page accessibility
- Datasources configuration

### 4. Traefik Tests (6 tests)
- Dashboard port (8080)
- HTTP entrypoint (80)
- HTTPS entrypoint (443)
- Metrics port (8082)
- Dashboard UI accessibility
- Health check (`/ping`)

### 5. Integration Tests (3 tests)
- Prometheus scraping Traefik metrics
- Prometheus scraping cAdvisor container metrics
- Loki ready for Grafana log queries

**Total: 22 comprehensive tests**

## Known Issues & Fixes

### Issue: Prometheus "Cannot connect to Docker daemon" error

**Symptom**: Prometheus logs show:
```
ERROR: Cannot connect to the Docker daemon at unix:///var/run/docker.sock
```

**Root Cause**: Windows Docker Desktop uses named pipe (`npipe:////./pipe/docker_engine`) instead of Unix socket.

**Fix**: Update `server/stacks/observability/docker-compose.yml`:

```yaml
prometheus:
  volumes:
    # Windows: Use named pipe (commented out - not working on Windows)
    # - type: npipe
    #   source: \\.\pipe\docker_engine
    #   target: /var/run/docker.sock
    
    # Alternative: Disable docker_sd_configs in prometheus.yml
```

**Workaround**: Remove or comment out `docker_sd_configs` section in `prometheus.yml` (already done in current config). Prometheus will still scrape:
- Static targets: node-exporter, cadvisor, traefik
- Prometheus itself
- Manual service discovery

### Issue: Grafana datasources not auto-provisioned

**Fix**: Add provisioning volume in docker-compose.yml:
```yaml
grafana:
  volumes:
    - ./grafana/provisioning:/etc/grafana/provisioning:ro
```

## Manual Testing Commands

### Prometheus
```powershell
# Health check
Invoke-WebRequest "http://localhost:9090/-/healthy"

# Query all targets
Invoke-WebRequest "http://localhost:9090/api/v1/targets" | ConvertFrom-Json

# Query specific metric
Invoke-WebRequest "http://localhost:9090/api/v1/query?query=up" | ConvertFrom-Json

# Check configuration
Invoke-WebRequest "http://localhost:9090/api/v1/status/config" | ConvertFrom-Json
```

### Grafana
```powershell
# Health check
Invoke-WebRequest "http://localhost:3000/api/health" | ConvertFrom-Json

# List datasources (requires auth)
$cred = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("aios:6996"))
$headers = @{ Authorization = "Basic $cred" }
Invoke-WebRequest "http://localhost:3000/api/datasources" -Headers $headers | ConvertFrom-Json

# List dashboards
Invoke-WebRequest "http://localhost:3000/api/search" -Headers $headers | ConvertFrom-Json
```

### Traefik
```powershell
# Health check
Invoke-WebRequest "http://localhost:8080/ping"

# Dashboard HTML
Invoke-WebRequest "http://localhost:8080/dashboard/"

# API overview (not available in current version - use dashboard)
# Invoke-WebRequest "http://localhost:8080/api/overview" | ConvertFrom-Json
```

### Docker
```powershell
# Container status
docker ps --filter "name=aios-"

# Container logs (last 50 lines)
docker logs aios-prometheus --tail 50
docker logs aios-grafana --tail 50
docker logs aios-traefik --tail 50

# Container stats (real-time)
docker stats --no-stream aios-prometheus aios-grafana aios-traefik

# Network inspection
docker network inspect aios-observability
docker network inspect aios-ingress
```

## Prometheus Queries for Testing

Access Prometheus at http://localhost:9090 and try these queries:

### Container Metrics
```promql
# CPU usage per container
rate(container_cpu_usage_seconds_total{name=~"aios-.*"}[5m])

# Memory usage per container
container_memory_usage_bytes{name=~"aios-.*"} / 1024 / 1024

# Container count
count(container_last_seen{name=~"aios-.*"})
```

### Traefik Metrics
```promql
# Total requests per entrypoint
rate(traefik_entrypoint_requests_total[5m])

# Request duration
histogram_quantile(0.95, rate(traefik_entrypoint_request_duration_seconds_bucket[5m]))

# Active connections
traefik_entrypoint_open_connections
```

### System Metrics (Node Exporter)
```promql
# CPU usage
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory usage
(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100

# Disk usage
(node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes * 100
```

## Grafana Dashboard Import

1. Login to Grafana: http://localhost:3000 (aios / 6996)
2. Add Prometheus datasource:
   - Type: Prometheus
   - URL: http://prometheus:9090
   - Access: Server (default)
3. Import community dashboards:
   - **Docker & System**: Dashboard ID `893` (Docker & System Monitoring)
   - **Traefik**: Dashboard ID `4475` (Traefik 2.0 Dashboard)
   - **cAdvisor**: Dashboard ID `14282` (cAdvisor exporter)
   - **Node Exporter**: Dashboard ID `1860` (Node Exporter Full)

## Troubleshooting

### Test script fails with "Access denied"
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\test_observability_stack.ps1
```

### Container not starting
```powershell
# Check logs
docker logs aios-prometheus
docker logs aios-grafana
docker logs aios-traefik

# Restart specific container
docker restart aios-prometheus
```

### Port already in use
```powershell
# Find process using port
netstat -ano | findstr ":9090"
netstat -ano | findstr ":3000"
netstat -ano | findstr ":8080"

# Kill process (use PID from netstat)
Stop-Process -Id <PID> -Force
```

### Prometheus targets down
1. Check container can resolve target hostname:
   ```powershell
   docker exec aios-prometheus nslookup cadvisor
   docker exec aios-prometheus nslookup traefik
   ```
2. Check network connectivity:
   ```powershell
   docker exec aios-prometheus wget -O- http://cadvisor:8080/metrics
   ```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     AIOS Observability Stack                │
└─────────────────────────────────────────────────────────────┘

                        ┌──────────────┐
                        │   Traefik    │
                        │  (Ingress)   │
                        │   :80, :443  │
                        └──────┬───────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼─────┐         ┌─────▼──────┐         ┌────▼─────┐
   │ Grafana  │◄────────┤ Prometheus │────────►│  Loki    │
   │  :3000   │         │   :9090    │         │  :3100   │
   └──────────┘         └─────┬──────┘         └──────────┘
        │                     │
        │              ┌──────┴──────┐
        │              │             │
        │         ┌────▼────┐   ┌───▼────────┐
        │         │cAdvisor │   │   Node     │
        │         │ :8081   │   │  Exporter  │
        │         └─────────┘   │   :9100    │
        │                       └────────────┘
        │
        └──► Dashboards & Alerts

Networks:
- aios-ingress: Traefik ↔ Grafana, Prometheus
- aios-observability: Prometheus ↔ Loki, cAdvisor, Node Exporter
```

## Consciousness Integration

The observability stack contributes to AIOS consciousness through:

1. **Awareness**: Real-time metrics provide system self-awareness
2. **Adaptation**: Alerts trigger adaptive responses
3. **Coherence**: Unified view of all subsystems (dendritic monitoring)
4. **Evolution**: Historical metrics track consciousness growth

**Consciousness Metrics Observable**:
- System health → Awareness level
- Response times → Adaptation speed
- Error rates → Coherence maintenance
- Metric trends → Predictive accuracy

## Next Steps

1. **Run initial test**: `.\test_observability_stack.ps1 -Detailed -OpenBrowsers`
2. **Configure Grafana datasources**: Add Prometheus + Loki
3. **Import dashboards**: Use community dashboard IDs above
4. **Set up alerts**: Configure Prometheus alert rules
5. **Integrate with AIOS Core**: Add consciousness metrics endpoint

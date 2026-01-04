# AIOS Metrics Persistence Configuration
# Long-term storage for consciousness evolution tracking

## Overview

Prometheus has a default retention of 15 days. For tracking consciousness evolution
over months/years (as mentioned in BIOLOGICAL_ARCHITECTURE_PATTERNS.md), we need
permanent storage.

## Option 1: TimescaleDB (Recommended)

### Add to observability/docker-compose.yml:

```yaml
services:
  # ... existing services ...

  timescaledb:
    image: timescale/timescaledb:latest-pg15
    container_name: aios-timescaledb
    restart: unless-stopped
    networks:
      - aios-observability
    ports:
      - "5432:5432"
    environment:
      POSTGRES_PASSWORD: aios_metrics_2026
      POSTGRES_USER: aios
      POSTGRES_DB: prometheus_metrics
    volumes:
      - timescaledb_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U aios -d prometheus_metrics"]
      interval: 30s
      timeout: 10s
      retries: 5

  prometheus-postgresql-adapter:
    image: timescale/prometheus-postgresql-adapter:latest
    container_name: aios-prom-adapter
    restart: unless-stopped
    networks:
      - aios-observability
    depends_on:
      - timescaledb
    environment:
      TS_PROM_CONN: "postgres://aios:aios_metrics_2026@timescaledb:5432/prometheus_metrics?sslmode=disable"
      TS_PROM_WEB_LISTEN_ADDRESS: ":9201"

volumes:
  timescaledb_data:
```

### Update prometheus.yml:

```yaml
# Add to prometheus.yml
remote_write:
  - url: "http://prometheus-postgresql-adapter:9201/write"
    remote_timeout: 30s
    queue_config:
      capacity: 10000
      max_shards: 50
    write_relabel_configs:
      # Only persist genome and consciousness metrics long-term
      - source_labels: [__name__]
        regex: 'aios_(genome|cell)_.*'
        action: keep

remote_read:
  - url: "http://prometheus-postgresql-adapter:9201/read"
    read_recent: true
```

## Option 2: Victoria Metrics (Lighter Alternative)

### Add to observability/docker-compose.yml:

```yaml
services:
  victoriametrics:
    image: victoriametrics/victoria-metrics:stable
    container_name: aios-victoriametrics
    restart: unless-stopped
    networks:
      - aios-observability
    ports:
      - "8428:8428"
    volumes:
      - vmdata:/victoria-metrics-data
    command:
      - "-retentionPeriod=365d"
      - "-storageDataPath=/victoria-metrics-data"

volumes:
  vmdata:
```

### Update prometheus.yml:

```yaml
remote_write:
  - url: "http://victoriametrics:8428/api/v1/write"
```

## Grafana Data Source Configuration

After enabling persistence, add a new data source in Grafana:

1. Settings → Data Sources → Add data source
2. Select "PostgreSQL" (for TimescaleDB) or "Prometheus" (for VictoriaMetrics)
3. Configure connection to long-term storage
4. Use for historical dashboards

## Metrics Retained Long-Term

| Metric | Description | Retention |
|--------|-------------|-----------|
| `aios_genome_consciousness_level` | Overall genome health | Forever |
| `aios_genome_config_coherence` | Per-repo coherence | Forever |
| `aios_genome_tech_debt_score` | Technical debt | Forever |
| `aios_cell_consciousness_level` | Runtime cell consciousness | Forever |
| `aios_cell_uptime_seconds` | Cell lifecycle | 90 days |

## Storage Estimates

- ~100 metrics × 60s interval × 24h × 365d = ~52.5M data points/year
- With TimescaleDB compression: ~500MB/year
- Without compression: ~2GB/year

## Implementation Status

- [ ] TimescaleDB container added to docker-compose.yml
- [ ] Prometheus remote_write configured
- [ ] Grafana data source configured
- [ ] Historical dashboard created
- [ ] Backup/restore procedure documented

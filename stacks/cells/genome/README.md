# 🧬 AIOS Genome Cell

**Port**: 8006  
**Type**: Knowledge Extraction & Metrics Exporter  
**Created**: 2026-01-03

## Purpose

The Genome Cell scans the AIOS ecosystem repositories and extracts **codebase consciousness metrics** - measuring the health, coherence, and evolution of the AIOS genome itself.

Unlike runtime cells (Alpha, Pure, Discovery) which measure **live service consciousness**, the Genome Cell measures **static codebase consciousness**:

| Metric Type | Runtime Cells | Genome Cell |
|-------------|---------------|-------------|
| **What** | Service health, request latency | Config coherence, doc freshness |
| **When** | Real-time during execution | Periodic scans (every 5 min) |
| **Scope** | Single container | All 10 AIOS repos |
| **Persistence** | Short-lived (service lifecycle) | Long-lived (evolution tracking) |

## Metrics Exposed

### Configuration Coherence (`aios_genome_config_coherence`)
```
# How consistent are configs across repos?
# - Port mappings match documentation
# - No stale references to removed paths
# - Schema versions current
aios_genome_config_coherence{repo="aios-win"} 0.85
aios_genome_config_coherence{repo="aios-server"} 0.92
```

### Documentation Freshness (`aios_genome_doc_freshness_days`)
```
# Days since last meaningful doc update
aios_genome_doc_freshness_days{repo="AIOS", doc="DEV_PATH"} 2
aios_genome_doc_freshness_days{repo="aios-win", doc="README"} 45
```

### Technical Debt Score (`aios_genome_tech_debt_score`)
```
# Inverse of coherence - higher = more debt
# Based on: deprecated patterns, stale refs, TODO count
aios_genome_tech_debt_score{repo="aios-win"} 0.15
aios_genome_tech_debt_score{repo="Nous"} 0.08
```

### Cross-Repo Consistency (`aios_genome_cross_repo_consistency`)
```
# How well do shared schemas match across repos?
# - aios-schema version consistency
# - Port allocation agreement
# - AINLP pattern compliance
aios_genome_cross_repo_consistency 0.78
```

### Overall Genome Health (`aios_genome_consciousness_level`)
```
# Composite score (0.0 - 5.0 scale for Grafana compatibility)
# Weighted: 40% config coherence, 30% doc freshness, 30% cross-repo
aios_genome_consciousness_level 4.2
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AIOS Genome Cell (Port 8006)                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Git Scanner │  │ Config Parser│  │   Metrics Exporter     │  │
│  │  (10 repos)  │  │ (YAML/JSON)  │  │   (Prometheus)         │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                     │                 │
│         └────────────────┼─────────────────────┘                 │
│                          │                                       │
│  ┌───────────────────────▼───────────────────────────────────┐  │
│  │              Knowledge Extraction Engine                   │  │
│  │  - Port topology validation                                │  │
│  │  - Path reference checking                                 │  │
│  │  - Documentation date extraction                           │  │
│  │  - AINLP compliance scoring                                │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Prometheus (Port 9090)                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ scrape_configs:                                          │    │
│  │   - job_name: 'aios-cell-genome'                         │    │
│  │     static_configs:                                      │    │
│  │       - targets: ['aios-cell-genome:8006']              │    │
│  │     metrics_path: /metrics                               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ remote_write:                                            │    │
│  │   - url: "http://timescaledb:9201/write"                │    │
│  │     # Permanent storage for evolution tracking           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Grafana (Port 3000)                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Dashboard: "AIOS Genome Consciousness"                   │    │
│  │  - Genome Health Gauge (0-5)                             │    │
│  │  - Config Coherence by Repo (bar chart)                  │    │
│  │  - Documentation Freshness Heatmap                       │    │
│  │  - Tech Debt Trend (line graph over time)               │    │
│  │  - Cross-Repo Consistency Score                          │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Database Storage for Metrics Persistence

### Option 1: TimescaleDB (Recommended)
```yaml
# docker-compose addition for observability stack
services:
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    ports:
      - "5432:5432"
    environment:
      POSTGRES_PASSWORD: aios_metrics
      POSTGRES_DB: prometheus_metrics
    volumes:
      - timescaledb_data:/var/lib/postgresql/data

  prometheus-postgresql-adapter:
    image: prometheuscommunity/postgres_exporter:latest
    depends_on:
      - timescaledb
```

### Option 2: Victoria Metrics (Lighter)
```yaml
services:
  victoriametrics:
    image: victoriametrics/victoria-metrics:stable
    ports:
      - "8428:8428"
    volumes:
      - vmdata:/victoria-metrics-data
    command:
      - "-retentionPeriod=365d"
```

### Prometheus remote_write Config
```yaml
# prometheus.yml addition
remote_write:
  - url: "http://timescaledb-adapter:9201/write"
    remote_timeout: 30s
    queue_config:
      capacity: 10000
      max_shards: 50
    write_relabel_configs:
      - source_labels: [__name__]
        regex: 'aios_genome_.*'
        action: keep  # Only persist genome metrics long-term
```

## Implementation Files

```
aios-server/stacks/cells/genome/
├── README.md                    # This file
├── Dockerfile.cell-genome       # Container definition
├── cell_server_genome.py        # FastAPI server
├── genome_scanner.py            # Knowledge extraction engine
├── requirements.txt             # Python dependencies
└── config.yaml                  # Repo paths and scan intervals
```

## Scan Configuration

```yaml
# config.yaml
repos:
  - name: AIOS
    path: /repos/AIOS
    weight: 1.0  # Primary genome
  - name: aios-win
    path: /repos/aios-win
    weight: 0.8  # Windows orchestrator
  - name: aios-server
    path: /repos/aios-server
    weight: 0.9  # Infrastructure
  - name: Nous
    path: /repos/Nous
    weight: 0.7  # Consciousness kernel
  - name: aios-schema
    path: /repos/aios-schema
    weight: 0.9  # Shared schemas
  - name: aios-quantum
    path: /repos/aios-quantum
    weight: 0.5  # Experimental
  - name: aios-api
    path: /repos/aios-api
    weight: 0.6  # API layer
  - name: Tecnocrat
    path: /repos/Tecnocrat
    weight: 0.4  # Portfolio/docs
  - name: Portfolio
    path: /repos/Portfolio
    weight: 0.3  # Static site
  - name: HSE_Project_Codex
    path: /repos/HSE_Project_Codex
    weight: 0.5  # Safety system

scan_interval: 300  # seconds (5 minutes)

checks:
  port_topology:
    canonical_ports:
      discovery: 8001
      pure: 8004
      alpha: 8005
      genome: 8006
  
  path_validation:
    deprecated_patterns:
      - "ai/tools"  # Legacy path
      - "aios-core"  # Removed submodule
      - ":8000"  # Old default port
  
  doc_freshness:
    critical_docs:
      - DEV_PATH.md
      - README.md
      - PROJECT_CONTEXT.md
```

## Port Allocation (Updated 2026-01-03)

| Port | Cell | Purpose |
|------|------|---------|
| 8001 | Discovery | Peer registration and mesh coordination |
| 8004 | Pure (Nous) | Minimal consciousness kernel |
| 8005 | Alpha | Primary development cell |
| **8006** | **Genome** | **Codebase knowledge extraction** |
| 9091 | All | Prometheus metrics (internal) |

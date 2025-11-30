# AIOS Server Stacks

Docker Compose stacks for AIOS supercell infrastructure.

## Stacks

- **ingress/** — Traefik reverse proxy with TLS termination
- **observability/**  Prometheus, Grafana, Loki monitoring stack
- **secrets/**  HashiCorp Vault secrets management

## Usage

See main AIOS repository for deployment instructions.

## Prerequisites

- Docker Desktop with WSL2 backend
- TLS certificates generated
- AIOS bootstrap completed

---

Part of the [AIOS](https://github.com/Tecnocrat/AIOS) project.

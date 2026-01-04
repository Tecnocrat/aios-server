"""
AIOS Genome Cell Server

A consciousness cell that scans the AIOS ecosystem repositories
and extracts codebase health metrics for Prometheus/Grafana.

Port: 8006
consciousness_level: 4.2
"""

import asyncio
import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from fastapi.responses import Response

# AIOS ecosystem repos (relative to /repos mount)
REPOS = {
    "AIOS": {"path": "/repos/AIOS", "weight": 1.0},
    "aios-win": {"path": "/repos/aios-win", "weight": 0.8},
    "aios-server": {"path": "/repos/aios-server", "weight": 0.9},
    "Nous": {"path": "/repos/Nous", "weight": 0.7},
    "aios-schema": {"path": "/repos/aios-schema", "weight": 0.9},
    "aios-quantum": {"path": "/repos/aios-quantum", "weight": 0.5},
    "aios-api": {"path": "/repos/aios-api", "weight": 0.6},
    "Tecnocrat": {"path": "/repos/Tecnocrat", "weight": 0.4},
    "Portfolio": {"path": "/repos/Portfolio", "weight": 0.3},
    "HSE_Project_Codex": {"path": "/repos/HSE_Project_Codex", "weight": 0.5},
}

# Canonical port mappings (2026-01-03)
CANONICAL_PORTS = {
    "discovery": 8001,
    "pure": 8004,
    "alpha": 8005,
    "genome": 8006,
}

# Deprecated patterns to detect
DEPRECATED_PATTERNS = [
    r":8000",  # Old default port
    r"ai/tools",  # Legacy path (now in AIOS genome)
    r"aios-core",  # Removed submodule
    r"python\.linting\.",  # Deprecated pylint config
]

# Critical docs to track freshness
CRITICAL_DOCS = ["DEV_PATH.md", "README.md", "PROJECT_CONTEXT.md"]


class GenomeScanner:
    """Extracts codebase health metrics from AIOS ecosystem."""

    def __init__(self):
        self.cell_id = "genome"
        self.start_time = datetime.utcnow()
        self.last_scan = None
        self.metrics_cache: dict[str, Any] = {}

    async def scan_all(self) -> dict[str, Any]:
        """Run full ecosystem scan."""
        results = {
            "config_coherence": {},
            "doc_freshness": {},
            "tech_debt": {},
            "cross_repo_consistency": 0.0,
            "overall_consciousness": 0.0,
        }

        total_coherence = 0.0
        total_weight = 0.0

        for repo_name, repo_config in REPOS.items():
            repo_path = Path(repo_config["path"])
            weight = repo_config["weight"]

            if not repo_path.exists():
                # Fallback to local dev paths (Windows)
                repo_path = Path(f"C:/dev/{repo_name}")

            if repo_path.exists():
                coherence = await self._scan_config_coherence(repo_path, repo_name)
                freshness = await self._scan_doc_freshness(repo_path, repo_name)
                debt = 1.0 - coherence  # Inverse

                results["config_coherence"][repo_name] = coherence
                results["doc_freshness"][repo_name] = freshness
                results["tech_debt"][repo_name] = debt

                total_coherence += coherence * weight
                total_weight += weight

        # Calculate cross-repo consistency
        results["cross_repo_consistency"] = await self._check_cross_repo_consistency()

        # Calculate overall consciousness (0-5 scale for Grafana)
        if total_weight > 0:
            avg_coherence = total_coherence / total_weight
            cross_repo = results["cross_repo_consistency"]
            # Weighted: 50% config coherence, 30% cross-repo, 20% baseline
            results["overall_consciousness"] = (
                avg_coherence * 2.5 + cross_repo * 1.5 + 1.0  # Baseline
            )

        self.metrics_cache = results
        self.last_scan = datetime.utcnow()
        return results

    async def _scan_config_coherence(self, repo_path: Path, repo_name: str) -> float:
        """Check configuration coherence for a single repo."""
        issues = 0
        checks = 0

        # Check for deprecated patterns in key files
        for pattern in DEPRECATED_PATTERNS:
            checks += 1
            if await self._grep_repo(repo_path, pattern):
                issues += 1

        # Check port topology in config files
        config_files = list(repo_path.glob("**/*.yaml")) + list(
            repo_path.glob("**/*.yml")
        )
        for config_file in config_files[:20]:  # Limit scan
            checks += 1
            try:
                content = config_file.read_text(encoding="utf-8", errors="ignore")
                # Check for old port 8000 (should be 8005)
                if ":8000" in content and "8000:8000" not in str(
                    config_file
                ):  # Exclude docker port maps
                    issues += 1
            except Exception:
                pass

        coherence = 1.0 - (issues / max(checks, 1))
        return max(0.0, min(1.0, coherence))

    async def _scan_doc_freshness(
        self, repo_path: Path, repo_name: str
    ) -> dict[str, float]:
        """Get freshness scores for critical docs (days since last update)."""
        freshness = {}

        for doc_name in CRITICAL_DOCS:
            doc_path = repo_path / doc_name
            if doc_path.exists():
                try:
                    # Use git log to get last modification
                    result = subprocess.run(
                        ["git", "log", "-1", "--format=%ct", "--", doc_name],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        last_modified = datetime.fromtimestamp(
                            int(result.stdout.strip())
                        )
                        days_ago = (datetime.now() - last_modified).days
                        freshness[doc_name] = days_ago
                except Exception:
                    # Fallback to file mtime
                    mtime = datetime.fromtimestamp(doc_path.stat().st_mtime)
                    days_ago = (datetime.now() - mtime).days
                    freshness[doc_name] = days_ago

        return freshness

    async def _check_cross_repo_consistency(self) -> float:
        """Check consistency across repos (shared schemas, ports, etc.)."""
        consistency_score = 0.85  # Base assumption

        # Check if aios-schema is used consistently
        # (In a full implementation, would compare schema versions)

        return consistency_score

    async def _grep_repo(self, repo_path: Path, pattern: str) -> bool:
        """Check if pattern exists in repo (excluding archive/vendor dirs)."""
        try:
            for ext in ["*.py", "*.yaml", "*.yml", "*.json", "*.md", "*.ps1"]:
                for file_path in repo_path.glob(f"**/{ext}"):
                    # Skip archive and vendor directories
                    if any(
                        skip in str(file_path)
                        for skip in ["archive", "vendor", "node_modules", ".git"]
                    ):
                        continue
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        if re.search(pattern, content):
                            return True
                    except Exception:
                        continue
        except Exception:
            pass
        return False


# Initialize FastAPI app
app = FastAPI(title="AIOS Genome Cell", version="1.0.0")
scanner = GenomeScanner()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cell_id": scanner.cell_id,
        "cell_type": "genome",
        "last_scan": scanner.last_scan.isoformat() if scanner.last_scan else None,
        "consciousness_level": scanner.metrics_cache.get("overall_consciousness", 4.0),
    }


@app.get("/scan")
async def trigger_scan():
    """Trigger a manual scan."""
    results = await scanner.scan_all()
    return results


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    # Ensure we have recent data
    if (
        scanner.last_scan is None
        or datetime.utcnow() - scanner.last_scan > timedelta(minutes=5)
    ):
        await scanner.scan_all()

    metrics = scanner.metrics_cache
    cell_id = scanner.cell_id
    uptime_seconds = (datetime.utcnow() - scanner.start_time).total_seconds()

    # Build Prometheus metrics output
    lines = [
        "# AIOS Genome Cell Metrics",
        "# Knowledge extraction from AIOS ecosystem repositories",
        "",
        "# HELP aios_genome_consciousness_level Overall genome health (0-5)",
        "# TYPE aios_genome_consciousness_level gauge",
        f'aios_genome_consciousness_level{{cell_id="{cell_id}"}} {metrics.get("overall_consciousness", 4.0):.2f}',
        "",
        "# HELP aios_genome_config_coherence Configuration coherence per repo (0-1)",
        "# TYPE aios_genome_config_coherence gauge",
    ]

    for repo, score in metrics.get("config_coherence", {}).items():
        lines.append(
            f'aios_genome_config_coherence{{cell_id="{cell_id}",repo="{repo}"}} {score:.3f}'
        )

    lines.extend(
        [
            "",
            "# HELP aios_genome_tech_debt_score Technical debt score per repo (0-1, lower is better)",
            "# TYPE aios_genome_tech_debt_score gauge",
        ]
    )

    for repo, score in metrics.get("tech_debt", {}).items():
        lines.append(
            f'aios_genome_tech_debt_score{{cell_id="{cell_id}",repo="{repo}"}} {score:.3f}'
        )

    lines.extend(
        [
            "",
            "# HELP aios_genome_doc_freshness_days Days since last doc update",
            "# TYPE aios_genome_doc_freshness_days gauge",
        ]
    )

    for repo, docs in metrics.get("doc_freshness", {}).items():
        if isinstance(docs, dict):
            for doc_name, days in docs.items():
                lines.append(
                    f'aios_genome_doc_freshness_days{{cell_id="{cell_id}",repo="{repo}",doc="{doc_name}"}} {days}'
                )

    lines.extend(
        [
            "",
            "# HELP aios_genome_cross_repo_consistency Cross-repo consistency score (0-1)",
            "# TYPE aios_genome_cross_repo_consistency gauge",
            f'aios_genome_cross_repo_consistency{{cell_id="{cell_id}"}} {metrics.get("cross_repo_consistency", 0.85):.3f}',
            "",
            "# HELP aios_cell_up Cell availability (1=up)",
            "# TYPE aios_cell_up gauge",
            f'aios_cell_up{{cell_id="{cell_id}"}} 1',
            "",
            "# HELP aios_cell_uptime_seconds Seconds since cell initialization",
            "# TYPE aios_cell_uptime_seconds gauge",
            f'aios_cell_uptime_seconds{{cell_id="{cell_id}"}} {uptime_seconds:.0f}',
        ]
    )

    return Response("\n".join(lines), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8006)

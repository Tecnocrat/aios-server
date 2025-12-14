#!/usr/bin/env python3
"""
AIOS Cell Birth Automation - Waypoint 26 (was 29)

Streamlined cell spawning using sibling repos pattern and aios-schema.

AINLP.bible[COMPLIANCE]: L3 (Enforced Dendritic Density)
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA INTEGRATION - Canonical types from aios-schema
# AINLP.dendritic[CONNECT] aios-schema → aios-server
#   Installation: pip install -e C:\dev\aios-schema
#   Pattern: Direct canonical import (no fallback - enforced dendritic density)
#   Pylance config: pyrightconfig.json extraPaths
#
# DENDRITIC ARCHITECTURE PRINCIPLE:
#   aios-schema is the single source of truth for cell types.
#   aios-server MUST have aios-schema installed - no standalone mode.
# ══════════════════════════════════════════════════════════════════════════════

from aios_schema import CellConfig, CellIdentity

# Schema verification (runtime assertion)
assert hasattr(CellConfig, 'name'), \
    "aios-schema CellConfig missing expected members"
assert hasattr(CellIdentity, 'to_dict'), \
    "aios-schema CellIdentity missing expected methods"


class CellBirther:
    """
    Automated cell birth orchestration.

    Spawns AIOS cells as Docker containers with proper configuration
    and mesh networking.
    """

    # Greek alphabet for cell naming
    CELL_NAMES = [
        "alpha", "beta", "gamma", "delta", "epsilon",
        "zeta", "eta", "theta", "iota", "kappa"
    ]

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.cells_dir = workspace_root / "aios-server" / "stacks" / "cells"
        self.registry_file = (
            workspace_root / "aios-win" / "config" / "cell_registry.json"
        )
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load or create cell registry."""
        if self.registry_file.exists():
            with open(self.registry_file, encoding='utf-8') as f:
                return json.load(f)
        return {"cells": {}, "next_port": 8001}

    def _save_registry(self):
        """Persist cell registry."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, "w", encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, default=str)

    def _get_next_cell_name(self) -> str:
        """Get next available cell name from Greek alphabet."""
        used = set(self.registry["cells"].keys())
        for name in self.CELL_NAMES:
            if name not in used:
                return name
        # Fallback to numbered cells
        i = len(used) + 1
        return f"cell-{i:02d}"

    def _get_next_port(self) -> int:
        """Get next available port."""
        port = self.registry["next_port"]
        self.registry["next_port"] = port + 1
        return port

    def birth(
        self,
        name: Optional[str] = None,
        port: Optional[int] = None,
        skip_core_build: bool = True,
        network: str = "aios-mesh"
    ) -> dict:
        """
        Birth a new AIOS cell.

        Args:
            name: Cell name (auto-assigned if None)
            port: Port to expose (auto-assigned if None)
            skip_core_build: Skip C++ native engine compilation
            network: Docker network to join

        Returns:
            Cell identity dict
        """
        name = name or self._get_next_cell_name()
        port = port or self._get_next_port()

        print(f"🧬 Birthing cell: {name} on port {port}")

        # Create cell config using canonical aios-schema types
        # AINLP.loader[latent:_config] Future cell orchestration/serialization
        _config = CellConfig(
            name=name,
            port=port,
            environment={
                "AIOS_CELL_ID": name,
                "AIOS_BRANCH": "main",
                "SKIP_CORE_BUILD": "1" if skip_core_build else "0",
                "PYTHONPATH": "/app"
            },
            networks=[network],
            skip_core_build=skip_core_build
        )
        # _config available for future use (cell orchestration, serialization)

        # Check if network exists, create if not
        self._ensure_network(network)

        # Build image if needed
        self._build_image(skip_core_build)

        # Run container
        container_name = f"aios-cell-{name}"
        env_args = [
            "-e", f"AIOS_CELL_ID={name}",
            "-e", "AIOS_BRANCH=main",
            "-e", f"SKIP_CORE_BUILD={'1' if skip_core_build else '0'}",
            "-e", "PYTHONPATH=/app"
        ]

        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--network", network,
            "-p", f"{port}:8000",
            "-p", f"{port + 1000}:9091",  # Metrics port
            *env_args,
            "--restart", "unless-stopped",
            "aios-cell:latest"
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )

        if result.returncode != 0:
            print(f"❌ Failed to birth cell: {result.stderr}")
            return {"error": result.stderr}

        container_id = result.stdout.strip()[:12]

        # Create identity using canonical aios-schema types
        identity = CellIdentity(
            name=name,
            host="localhost",
            port=port,
            version="0.1.0"
        )
        identity_dict = identity.to_dict()
        # Add runtime metadata
        identity_dict["birth_time"] = datetime.now().isoformat()
        identity_dict["container_id"] = container_id

        # Update registry
        self.registry["cells"][name] = {
            **identity_dict,
            "container_id": container_id,
            "status": "healthy"
        }
        self._save_registry()

        print(f"✅ Cell {name} born: http://localhost:{port}")
        print(f"   Container: {container_id}")
        print(f"   Metrics: http://localhost:{port + 1000}")

        return identity_dict

    def _ensure_network(self, network: str):
        """Ensure Docker network exists."""
        result = subprocess.run(
            ["docker", "network", "inspect", network],
            capture_output=True,
            check=False
        )
        if result.returncode != 0:
            print(f"🔗 Creating network: {network}")
            subprocess.run(
                ["docker", "network", "create", network],
                check=False
            )

    def _build_image(self, _skip_core_build: bool):
        """Build cell image if needed."""
        # Note: _skip_core_build reserved for future use (native core)
        # Check if image exists
        result = subprocess.run(
            ["docker", "images", "-q", "aios-cell:latest"],
            capture_output=True,
            text=True,
            check=False
        )

        if not result.stdout.strip():
            print("🔨 Building cell image...")
            dockerfile = self.cells_dir / "beta" / "Dockerfile.cell"

            if not dockerfile.exists():
                print(f"❌ Dockerfile not found: {dockerfile}")
                return

            # Build from workspace root
            cmd = [
                "docker", "build",
                "-f", str(dockerfile),
                "-t", "aios-cell:latest",
                str(self.workspace_root / "aios-win")  # Build context
            ]
            subprocess.run(cmd, cwd=self.workspace_root, check=False)

    def list_cells(self) -> list:
        """List all registered cells."""
        cells = []
        for name, info in self.registry["cells"].items():
            # Check actual status
            result = subprocess.run(
                ["docker", "inspect", f"aios-cell-{name}"],
                capture_output=True,
                check=False
            )
            status = "running" if result.returncode == 0 else "stopped"
            cells.append({
                "name": name,
                "port": info.get("port"),
                "status": status,
                "container_id": info.get("container_id", "unknown")
            })
        return cells

    def kill(self, name: str) -> bool:
        """Kill a cell (stop and remove container)."""
        container_name = f"aios-cell-{name}"

        # Stop
        subprocess.run(
            ["docker", "stop", container_name],
            capture_output=True,
            check=False
        )
        # Remove
        result = subprocess.run(
            ["docker", "rm", container_name],
            capture_output=True,
            check=False
        )

        if result.returncode == 0:
            if name in self.registry["cells"]:
                del self.registry["cells"][name]
                self._save_registry()
            print(f"💀 Cell {name} terminated")
            return True
        return False


def main():
    parser = argparse.ArgumentParser(
        description="AIOS Cell Birth Automation"
    )
    parser.add_argument("action", choices=["birth", "list", "kill"])
    parser.add_argument("--name", help="Cell name (auto if not provided)")
    parser.add_argument("--port", type=int, help="Port to expose")
    parser.add_argument(
        "--with-core", action="store_true", help="Build C++ core"
    )

    args = parser.parse_args()

    # Find workspace root (where all sibling repos are)
    workspace_root = Path(__file__).parent.parent.parent.parent

    birther = CellBirther(workspace_root)

    if args.action == "birth":
        result = birther.birth(
            name=args.name,
            port=args.port,
            skip_core_build=not args.with_core
        )
        print(json.dumps(result, indent=2, default=str))

    elif args.action == "list":
        cells = birther.list_cells()
        if cells:
            hdr = f"{'Name':<10} {'Port':<8} {'Status':<10} {'Container':<15}"
            print(f"\n{hdr}")
            print("-" * 45)
            for cell in cells:
                row = (f"{cell['name']:<10} {cell['port']:<8} "
                       f"{cell['status']:<10} {cell['container_id']:<15}")
                print(row)
        else:
            print("No cells registered")

    elif args.action == "kill":
        if not args.name:
            print("❌ --name required for kill action")
            sys.exit(1)
        birther.kill(args.name)


if __name__ == "__main__":
    main()

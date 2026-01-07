#!/usr/bin/env python3
"""
AIOS Organism-001 Backup & Recovery Manager

Comprehensive backup system for SimplCell metadata:
- Exports all cell state, conversations, and memory to JSON
- Creates timestamped backups with integrity verification
- Supports restore from backup after hard reset
- Maintains backup rotation policy

Usage:
    python organism_backup.py backup        # Create full backup
    python organism_backup.py restore       # Restore from latest backup
    python organism_backup.py status        # Check backup status
    python organism_backup.py list          # List available backups
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import asyncio

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CELLS = [
    {"name": "simplcell-alpha", "port": 8900},
    {"name": "simplcell-beta", "port": 8901},
]

BACKUP_DIR = Path(__file__).parent / "backups" / "organism-001"
BACKUP_RETENTION = 20  # Keep last N backups


# ═══════════════════════════════════════════════════════════════════════════════
# BACKUP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_cell_metadata(cell: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fetch complete metadata from a cell."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://localhost:{cell['port']}/metadata",
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"  ✅ {cell['name']}: {len(data.get('conversations', []))} conversations")
                    return data
                else:
                    print(f"  ⚠️ {cell['name']}: HTTP {resp.status}")
                    return None
    except Exception as e:
        print(f"  ❌ {cell['name']}: {e}")
        return None


def calculate_checksum(data: Dict[str, Any]) -> str:
    """Calculate SHA256 checksum of backup data."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


async def create_backup() -> Optional[Path]:
    """Create a comprehensive backup of all cells."""
    print("\n🔄 Creating Organism-001 backup...")
    
    # Ensure backup directory exists
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect metadata from all cells
    backup_data = {
        "organism_id": "organism-001",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backup_version": "1.0",
        "cells": {}
    }
    
    print("\n📡 Fetching cell metadata...")
    for cell in CELLS:
        metadata = await fetch_cell_metadata(cell)
        if metadata:
            backup_data["cells"][cell["name"]] = metadata
    
    if not backup_data["cells"]:
        print("\n❌ No cells available for backup")
        return None
    
    # Calculate integrity checksum
    backup_data["checksum"] = calculate_checksum(backup_data["cells"])
    
    # Create timestamped backup file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = BACKUP_DIR / f"organism-001_{timestamp}.json"
    
    # Write backup
    with open(backup_file, "w", encoding="utf-8") as f:
        json.dump(backup_data, f, indent=2, default=str)
    
    # Create symlink to latest
    latest_link = BACKUP_DIR / "latest.json"
    if latest_link.exists():
        latest_link.unlink()
    # Copy instead of symlink for Windows compatibility
    shutil.copy2(backup_file, latest_link)
    
    # Calculate stats
    total_conversations = sum(
        len(cell.get("conversations", []))
        for cell in backup_data["cells"].values()
    )
    total_memory = sum(
        len(cell.get("memory_buffer", []))
        for cell in backup_data["cells"].values()
    )
    file_size = backup_file.stat().st_size
    
    print(f"\n✅ Backup created: {backup_file.name}")
    print(f"   📊 Cells: {len(backup_data['cells'])}")
    print(f"   💬 Total conversations: {total_conversations}")
    print(f"   🧠 Total memory entries: {total_memory}")
    print(f"   📦 File size: {file_size / 1024:.1f} KB")
    print(f"   🔐 Checksum: {backup_data['checksum']}")
    
    # Cleanup old backups
    cleanup_old_backups()
    
    return backup_file


def cleanup_old_backups():
    """Remove backups beyond retention limit."""
    backups = sorted(BACKUP_DIR.glob("organism-001_*.json"))
    while len(backups) > BACKUP_RETENTION:
        old_backup = backups.pop(0)
        old_backup.unlink()
        print(f"   🗑️ Removed old backup: {old_backup.name}")


def list_backups() -> List[Dict[str, Any]]:
    """List all available backups."""
    print("\n📚 Available Organism-001 Backups:\n")
    
    if not BACKUP_DIR.exists():
        print("   No backups found.")
        return []
    
    backups = sorted(BACKUP_DIR.glob("organism-001_*.json"), reverse=True)
    
    if not backups:
        print("   No backups found.")
        return []
    
    results = []
    for backup in backups:
        try:
            with open(backup, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            total_conv = sum(
                len(cell.get("conversations", []))
                for cell in data.get("cells", {}).values()
            )
            
            size = backup.stat().st_size / 1024
            
            results.append({
                "file": backup.name,
                "created": data.get("created_at", "unknown"),
                "cells": len(data.get("cells", {})),
                "conversations": total_conv,
                "size_kb": size,
                "checksum": data.get("checksum", "N/A")
            })
            
            print(f"   📦 {backup.name}")
            print(f"      Created: {data.get('created_at', 'unknown')}")
            print(f"      Cells: {len(data.get('cells', {}))} | Conversations: {total_conv}")
            print(f"      Size: {size:.1f} KB | Checksum: {data.get('checksum', 'N/A')[:8]}...")
            print()
            
        except Exception as e:
            print(f"   ⚠️ {backup.name}: Error reading - {e}")
    
    return results


def show_status():
    """Show current backup status."""
    print("\n📊 Organism-001 Backup Status\n")
    
    if not BACKUP_DIR.exists():
        print("   ❌ No backup directory found")
        print(f"      Expected: {BACKUP_DIR}")
        return
    
    backups = list(BACKUP_DIR.glob("organism-001_*.json"))
    latest = BACKUP_DIR / "latest.json"
    
    print(f"   📁 Backup directory: {BACKUP_DIR}")
    print(f"   📚 Total backups: {len(backups)}")
    print(f"   🔄 Retention policy: {BACKUP_RETENTION} backups")
    
    if latest.exists():
        try:
            with open(latest, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            total_conv = sum(
                len(cell.get("conversations", []))
                for cell in data.get("cells", {}).values()
            )
            
            print(f"\n   📌 Latest backup:")
            print(f"      Created: {data.get('created_at', 'unknown')}")
            print(f"      Cells: {len(data.get('cells', {}))}")
            print(f"      Conversations: {total_conv}")
            print(f"      Checksum: {data.get('checksum', 'N/A')}")
        except Exception as e:
            print(f"\n   ⚠️ Cannot read latest backup: {e}")
    else:
        print("\n   ❌ No 'latest.json' symlink found")


async def restore_backup(backup_file: Optional[str] = None):
    """
    Restore from backup.
    
    Note: This generates a restore script since direct database restore
    requires the containers to be stopped or use import endpoints.
    """
    print("\n♻️ Organism-001 Restore Process\n")
    
    # Find backup file
    if backup_file:
        path = BACKUP_DIR / backup_file
    else:
        path = BACKUP_DIR / "latest.json"
    
    if not path.exists():
        print(f"   ❌ Backup not found: {path}")
        return
    
    print(f"   📂 Restoring from: {path.name}")
    
    # Load backup
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Verify checksum
    expected = data.get("checksum")
    actual = calculate_checksum(data.get("cells", {}))
    
    if expected != actual:
        print(f"   ⚠️ Checksum mismatch! Expected {expected}, got {actual}")
        print("   Backup may be corrupted.")
        return
    
    print(f"   ✅ Checksum verified: {expected}")
    
    # Generate restore files
    restore_dir = BACKUP_DIR / "restore_staging"
    restore_dir.mkdir(exist_ok=True)
    
    for cell_name, cell_data in data.get("cells", {}).items():
        cell_file = restore_dir / f"{cell_name}_restore.json"
        with open(cell_file, "w", encoding="utf-8") as f:
            json.dump(cell_data, f, indent=2, default=str)
        print(f"   📄 Created: {cell_file.name}")
    
    print(f"\n   📁 Restore files staged in: {restore_dir}")
    print("\n   To complete restore:")
    print("   1. Stop SimplCell containers: docker compose down")
    print("   2. Copy database files from Docker volumes")
    print("   3. Use import_backup.py to load conversations")
    print("   4. Restart containers: docker compose up -d")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Organism-001 Backup & Recovery Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python organism_backup.py backup     Create full backup
    python organism_backup.py status     Check backup status  
    python organism_backup.py list       List all backups
    python organism_backup.py restore    Restore from latest backup
        """
    )
    
    parser.add_argument(
        "command",
        choices=["backup", "restore", "status", "list"],
        help="Action to perform"
    )
    parser.add_argument(
        "--file", "-f",
        help="Specific backup file for restore"
    )
    
    args = parser.parse_args()
    
    if args.command == "backup":
        asyncio.run(create_backup())
    elif args.command == "restore":
        asyncio.run(restore_backup(args.file))
    elif args.command == "status":
        show_status()
    elif args.command == "list":
        list_backups()


if __name__ == "__main__":
    main()

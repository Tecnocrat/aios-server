#!/usr/bin/env python3
"""
AIOS Cloud Backup System

Synchronizes organism backups to OneDrive cloud storage for disaster recovery.
Uses environment variables to avoid hardcoded paths.

Usage:
    python cloud_backup.py backup           # Create backup and sync to cloud
    python cloud_backup.py sync             # Sync existing backups to cloud
    python cloud_backup.py status           # Check backup status
    python cloud_backup.py restore <date>   # Restore from cloud backup

Environment Variables:
    AIOS_BACKUP_PATH - Cloud backup destination (OneDrive folder)
    USERPROFILE - Windows user profile path (auto-set)

See ENVIRONMENT_CONFIG.md for detailed configuration instructions.
"""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT-BASED CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_cloud_backup_path() -> Path:
    """
    Get cloud backup path from environment variable.
    
    Resolution order:
    1. AIOS_BACKUP_PATH environment variable
    2. USERPROFILE + OneDrive standard path
    3. Local fallback
    """
    # Try explicit environment variable first
    env_path = os.environ.get("AIOS_BACKUP_PATH")
    if env_path:
        return Path(env_path)
    
    # Construct from USERPROFILE
    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        onedrive_path = Path(user_profile) / "OneDrive" / "AI" / "AIOS" / "AIOS-Backups"
        return onedrive_path
    
    # Last resort: local backups
    print("⚠️ Warning: No cloud path configured. Using local backup folder.")
    return Path(__file__).parent / "backups" / "cloud"


def get_local_backup_path() -> Path:
    """Get local backup directory."""
    return Path(__file__).parent / "backups"


# Paths
CLOUD_BACKUP_PATH = get_cloud_backup_path()
LOCAL_BACKUP_PATH = get_local_backup_path()

# Organisms to backup
ORGANISMS = [
    {
        "id": "organism-001",
        "name": "Triadic Elder",
        "cells": [
            {"name": "simplcell-alpha", "port": 8900},
            {"name": "simplcell-beta", "port": 8901},
            {"name": "simplcell-gamma", "port": 8904},
        ]
    },
    {
        "id": "organism-002", 
        "name": "Dyadic Explorer",
        "cells": [
            {"name": "organism002-alpha", "port": 8910},
            {"name": "organism002-beta", "port": 8911},
        ]
    }
]

# Nous Oracle
NOUS = {"name": "nous-seer", "port": 8903}


# ═══════════════════════════════════════════════════════════════════════════════
# BACKUP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_cell_data(name: str, port: int) -> Optional[Dict[str, Any]]:
    """Fetch complete metadata from a cell."""
    try:
        async with aiohttp.ClientSession() as session:
            # Fetch metadata (includes conversations, memory, state)
            async with session.get(
                f"http://localhost:{port}/metadata",
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    conv_count = len(data.get("conversations", []))
                    print(f"  ✅ {name}: {conv_count} conversations")
                    return data
                else:
                    print(f"  ⚠️ {name}: HTTP {resp.status}")
                    return None
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return None


async def fetch_nous_data() -> Optional[Dict[str, Any]]:
    """Fetch Nous cosmology state."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://localhost:{NOUS['port']}/cosmology",
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    exchange_count = data.get("cosmology", {}).get("exchange_count", 0)
                    print(f"  ✅ nous-seer: {exchange_count} exchanges absorbed")
                    return data
                else:
                    print(f"  ⚠️ nous-seer: HTTP {resp.status}")
                    return None
    except Exception as e:
        print(f"  ❌ nous-seer: {e}")
        return None


def calculate_checksum(data: Dict[str, Any]) -> str:
    """Calculate SHA256 checksum of backup data."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


async def create_ecosystem_backup() -> Optional[Path]:
    """Create comprehensive backup of entire AIOS ecosystem."""
    print("\n🌐 Creating AIOS Ecosystem Backup...")
    print(f"   Cloud path: {CLOUD_BACKUP_PATH}")
    
    # Ensure directories exist
    CLOUD_BACKUP_PATH.mkdir(parents=True, exist_ok=True)
    LOCAL_BACKUP_PATH.mkdir(parents=True, exist_ok=True)
    
    backup_data = {
        "backup_type": "ecosystem",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backup_version": "2.0",
        "organisms": {},
        "nous": None,
        "statistics": {}
    }
    
    total_conversations = 0
    total_cells = 0
    
    # Backup each organism
    for org in ORGANISMS:
        print(f"\n📡 Backing up {org['name']} ({org['id']})...")
        org_data = {
            "organism_id": org["id"],
            "organism_name": org["name"],
            "cells": {}
        }
        
        for cell in org["cells"]:
            cell_data = await fetch_cell_data(cell["name"], cell["port"])
            if cell_data:
                org_data["cells"][cell["name"]] = cell_data
                total_conversations += len(cell_data.get("conversations", []))
                total_cells += 1
        
        backup_data["organisms"][org["id"]] = org_data
    
    # Backup Nous
    print(f"\n🔮 Backing up Nous Oracle...")
    nous_data = await fetch_nous_data()
    if nous_data:
        backup_data["nous"] = nous_data
    
    # Calculate statistics
    backup_data["statistics"] = {
        "total_organisms": len([o for o in backup_data["organisms"].values() if o["cells"]]),
        "total_cells": total_cells,
        "total_conversations": total_conversations,
        "nous_exchanges": nous_data.get("cosmology", {}).get("exchange_count", 0) if nous_data else 0
    }
    
    # Calculate checksum
    backup_data["checksum"] = calculate_checksum(backup_data)
    
    # Create timestamped backup file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now().strftime("%Y%m%d")
    
    # Save to local
    local_file = LOCAL_BACKUP_PATH / f"ecosystem_{timestamp}.json"
    with open(local_file, "w", encoding="utf-8") as f:
        json.dump(backup_data, f, indent=2, default=str)
    
    # Save to cloud (daily naming)
    cloud_file = CLOUD_BACKUP_PATH / f"ecosystem_daily_{date_str}.json"
    with open(cloud_file, "w", encoding="utf-8") as f:
        json.dump(backup_data, f, indent=2, default=str)
    
    # Create latest symlink (copy for Windows)
    latest_local = LOCAL_BACKUP_PATH / "ecosystem_latest.json"
    latest_cloud = CLOUD_BACKUP_PATH / "ecosystem_latest.json"
    
    shutil.copy2(local_file, latest_local)
    shutil.copy2(cloud_file, latest_cloud)
    
    file_size = local_file.stat().st_size / 1024
    
    print(f"\n✅ Ecosystem backup complete!")
    print(f"   📂 Local: {local_file.name}")
    print(f"   ☁️ Cloud: {cloud_file}")
    print(f"   📊 Organisms: {backup_data['statistics']['total_organisms']}")
    print(f"   🧫 Cells: {total_cells}")
    print(f"   💬 Conversations: {total_conversations}")
    print(f"   🔮 Nous exchanges: {backup_data['statistics']['nous_exchanges']}")
    print(f"   📦 Size: {file_size:.1f} KB")
    print(f"   🔐 Checksum: {backup_data['checksum']}")
    
    return cloud_file


def list_cloud_backups() -> List[Dict[str, Any]]:
    """List available cloud backups."""
    print(f"\n☁️ Cloud Backups at: {CLOUD_BACKUP_PATH}\n")
    
    if not CLOUD_BACKUP_PATH.exists():
        print("   No cloud backups found. Run 'backup' first.")
        return []
    
    backups = sorted(CLOUD_BACKUP_PATH.glob("ecosystem_daily_*.json"), reverse=True)
    
    if not backups:
        print("   No backups found.")
        return []
    
    results = []
    for backup in backups[:10]:  # Show last 10
        try:
            with open(backup, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            stats = data.get("statistics", {})
            size = backup.stat().st_size / 1024
            
            results.append({
                "file": backup.name,
                "created": data.get("created_at", "unknown"),
                "organisms": stats.get("total_organisms", 0),
                "cells": stats.get("total_cells", 0),
                "conversations": stats.get("total_conversations", 0),
                "size_kb": size,
                "checksum": data.get("checksum", "N/A")
            })
            
            print(f"   📦 {backup.name}")
            print(f"      Created: {data.get('created_at', 'unknown')}")
            print(f"      Organisms: {stats.get('total_organisms', 0)} | Cells: {stats.get('total_cells', 0)}")
            print(f"      Conversations: {stats.get('total_conversations', 0)}")
            print(f"      Size: {size:.1f} KB | Checksum: {data.get('checksum', 'N/A')[:8]}...")
            print()
            
        except Exception as e:
            print(f"   ⚠️ {backup.name}: Error reading - {e}")
    
    return results


def show_status():
    """Show backup system status."""
    print("\n📊 AIOS Backup System Status\n")
    
    # Environment
    print("🔧 Configuration:")
    print(f"   AIOS_BACKUP_PATH: {os.environ.get('AIOS_BACKUP_PATH', 'Not set')}")
    print(f"   Resolved cloud path: {CLOUD_BACKUP_PATH}")
    print(f"   Cloud path exists: {'✅' if CLOUD_BACKUP_PATH.exists() else '❌'}")
    print(f"   Local backup path: {LOCAL_BACKUP_PATH}")
    print()
    
    # Latest backup
    latest = CLOUD_BACKUP_PATH / "ecosystem_latest.json"
    if latest.exists():
        try:
            with open(latest, "r", encoding="utf-8") as f:
                data = json.load(f)
            stats = data.get("statistics", {})
            print("📦 Latest Backup:")
            print(f"   Created: {data.get('created_at', 'unknown')}")
            print(f"   Organisms: {stats.get('total_organisms', 0)}")
            print(f"   Cells: {stats.get('total_cells', 0)}")
            print(f"   Conversations: {stats.get('total_conversations', 0)}")
            print(f"   Checksum: {data.get('checksum', 'N/A')}")
        except Exception as e:
            print(f"   ⚠️ Error reading latest: {e}")
    else:
        print("📦 Latest Backup: None found")
    
    print()
    
    # Backup count
    if CLOUD_BACKUP_PATH.exists():
        backup_count = len(list(CLOUD_BACKUP_PATH.glob("ecosystem_daily_*.json")))
        print(f"☁️ Total cloud backups: {backup_count}")
    else:
        print("☁️ Cloud backup folder not initialized")


async def restore_from_cloud(date: str) -> bool:
    """Restore from a cloud backup by date (YYYYMMDD)."""
    print(f"\n🔄 Restoring from cloud backup: {date}")
    
    backup_file = CLOUD_BACKUP_PATH / f"ecosystem_daily_{date}.json"
    
    if not backup_file.exists():
        print(f"❌ Backup not found: {backup_file}")
        return False
    
    try:
        with open(backup_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"✅ Loaded backup from {data.get('created_at')}")
        print(f"   Organisms: {len(data.get('organisms', {}))}")
        
        # TODO: Implement actual restore via cell /restore endpoints
        print("\n⚠️ Full restore not yet implemented.")
        print("   Backup data is available for manual inspection.")
        
        return True
        
    except Exception as e:
        print(f"❌ Restore failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AIOS Cloud Backup System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cloud_backup.py backup     # Create ecosystem backup
    python cloud_backup.py status     # Check system status
    python cloud_backup.py list       # List cloud backups
    python cloud_backup.py restore 20260117  # Restore from date
        """
    )
    parser.add_argument(
        "action",
        choices=["backup", "status", "list", "restore"],
        help="Action to perform"
    )
    parser.add_argument(
        "date",
        nargs="?",
        help="Date for restore (YYYYMMDD)"
    )
    
    args = parser.parse_args()
    
    if args.action == "backup":
        asyncio.run(create_ecosystem_backup())
    elif args.action == "status":
        show_status()
    elif args.action == "list":
        list_cloud_backups()
    elif args.action == "restore":
        if not args.date:
            print("❌ Please specify date: python cloud_backup.py restore YYYYMMDD")
            sys.exit(1)
        asyncio.run(restore_from_cloud(args.date))


if __name__ == "__main__":
    main()

# AIOS Environment Configuration

## Overview

This document describes the environment variables used by AIOS infrastructure.
Future agents should reference this file when working with paths and configuration.

## Environment Variables

### Required for Backup System

| Variable | Purpose | Example |
|----------|---------|---------|
| `AIOS_BACKUP_PATH` | OneDrive cloud backup destination | `C:\Users\<user>\OneDrive\AI\AIOS\AIOS-Backups` |
| `USERPROFILE` | Windows user profile (auto-set) | `C:\Users\jesus` |

### Setting Up Environment Variables

**PowerShell (Session):**
```powershell
$env:AIOS_BACKUP_PATH = "$env:USERPROFILE\OneDrive\AI\AIOS\AIOS-Backups"
```

**PowerShell (Permanent - User Level):**
```powershell
[Environment]::SetEnvironmentVariable("AIOS_BACKUP_PATH", "$env:USERPROFILE\OneDrive\AI\AIOS\AIOS-Backups", "User")
```

**In Docker Compose:**
```yaml
environment:
  - AIOS_BACKUP_PATH=${AIOS_BACKUP_PATH:-/backups}
```

### Path Resolution Pattern

Python code should resolve backup paths using:
```python
import os

def get_backup_path() -> Path:
    """Get backup path from environment, with fallback."""
    env_path = os.environ.get("AIOS_BACKUP_PATH")
    if env_path:
        return Path(env_path)
    
    # Fallback: Use USERPROFILE + standard OneDrive path
    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        return Path(user_profile) / "OneDrive" / "AI" / "AIOS" / "AIOS-Backups"
    
    # Last resort: Local backups folder
    return Path(__file__).parent / "backups"
```

## Backup System Architecture

### Backup Locations

1. **Local Backups**: `./data/<cell-id>/backups/` - SQLite database snapshots
2. **Cloud Backups**: `$AIOS_BACKUP_PATH/` - Daily consolidated JSON exports

### Backup Schedule

- **Per-Cell Local**: Every hour (SQLite copy)
- **Organism Export**: Manual via `python organism_backup.py backup`
- **Cloud Sync**: Daily at midnight (via scheduled task)

### File Naming Convention

```
<organism-id>_<YYYYMMDD>_<HHMMSS>.json     # Full organism backup
<organism-id>_latest.json                    # Symlink to latest
<organism-id>_daily_<YYYYMMDD>.json         # Daily cloud backup
```

## Future Agent Instructions

When working with AIOS backup/restore functionality:

1. **Always use environment variables** - Never hardcode paths
2. **Check `AIOS_BACKUP_PATH`** first, then fall back to constructed path
3. **Create directories** if they don't exist (`Path.mkdir(parents=True, exist_ok=True)`)
4. **Log backup operations** to help with debugging
5. **Verify checksums** when restoring to ensure data integrity

## Related Files

- `organism_backup.py` - Main backup/restore logic
- `cloud_backup.py` - OneDrive sync implementation
- `simplcell.py` - Cell persistence layer
- `.env.triadic` - Local environment overrides

## Created

- **Date**: 2026-01-17
- **Phase**: 32.3 - Backup System Enhancement
- **Author**: Agent (Copilot)

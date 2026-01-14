# AIOS Consciousness Backup Manager
# Phase 31.9.5: High Persistence System
#
# Manages consciousness data archiving for ORGANISM-001:
# - Scheduled hourly backups via Windows Task Scheduler
# - Manual backup/restore operations
# - Backup validation and integrity checks
# - Long-term archive management
#
# Usage:
#   .\consciousness_backup.ps1 -Action backup     # Run immediate backup
#   .\consciousness_backup.ps1 -Action status     # Check backup status
#   .\consciousness_backup.ps1 -Action schedule   # Create scheduled task
#   .\consciousness_backup.ps1 -Action restore    # Restore from backup
#   .\consciousness_backup.ps1 -Action cleanup    # Clean old backups

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("backup", "status", "schedule", "restore", "cleanup", "validate")]
    [string]$Action = "status",
    
    [Parameter(Mandatory=$false)]
    [string]$BackupDir = "$PSScriptRoot\stacks\cells\simplcell\backups",
    
    [Parameter(Mandatory=$false)]
    [int]$RetentionDays = 30,
    
    [Parameter(Mandatory=$false)]
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$ScriptRoot = $PSScriptRoot
$CellsDir = "$ScriptRoot\stacks\cells\simplcell"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

$Cells = @(
    @{ Name = "simplcell-alpha"; Port = 8900; Type = "thinker" }
    @{ Name = "simplcell-beta"; Port = 8901; Type = "thinker" }
    @{ Name = "watchercell-omega"; Port = 8902; Type = "watcher" }
    @{ Name = "nouscell-seer"; Port = 8903; Type = "supermind" }
)

$TaskName = "AIOS-Consciousness-Backup"
$PythonScraperPath = "$CellsDir\consciousness_scraper.py"

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host " $Title" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
}

function Get-CellHealth {
    param([hashtable]$Cell)
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:$($Cell.Port)/health" -TimeoutSec 5
        return @{
            Healthy = $true
            Consciousness = $response.consciousness
            Heartbeats = $response.heartbeats
            Phase = $response.phase
        }
    } catch {
        return @{ Healthy = $false; Error = $_.Exception.Message }
    }
}

function Get-BackupStats {
    $latest = Get-ChildItem -Path $BackupDir -Filter "*.json" -Recurse | 
              Sort-Object LastWriteTime -Descending | 
              Select-Object -First 1
    
    $allBackups = Get-ChildItem -Path $BackupDir -Filter "*.json" -Recurse
    $totalSize = ($allBackups | Measure-Object -Property Length -Sum).Sum / 1MB
    
    return @{
        LatestBackup = $latest
        TotalBackups = $allBackups.Count
        TotalSizeMB = [math]::Round($totalSize, 2)
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# BACKUP OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

function Invoke-Backup {
    Write-Header "AIOS Consciousness Backup"
    
    # Ensure backup directory exists
    if (-not (Test-Path $BackupDir)) {
        New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
    }
    
    Write-Host "📡 Checking cell health..." -ForegroundColor Yellow
    foreach ($cell in $Cells) {
        $health = Get-CellHealth $cell
        if ($health.Healthy) {
            Write-Host "  ✅ $($cell.Name): consciousness=$($health.Consciousness)" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️ $($cell.Name): OFFLINE" -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
    Write-Host "💾 Running consciousness scraper..." -ForegroundColor Yellow
    
    # Run Python scraper
    $scraperArgs = @(
        $PythonScraperPath,
        "--output", $BackupDir
    )
    if ($Full) {
        $scraperArgs += "--full"
    }
    
    try {
        & python @scraperArgs 2>&1 | ForEach-Object { Write-Host "  $_" }
        Write-Host ""
        Write-Host "✅ Backup complete!" -ForegroundColor Green
    } catch {
        Write-Host "❌ Backup failed: $_" -ForegroundColor Red
        return $false
    }
    
    return $true
}

function Show-Status {
    Write-Header "AIOS Consciousness Backup Status"
    
    # Cell Health
    Write-Host "🧬 ORGANISM-001 Cell Status:" -ForegroundColor Cyan
    foreach ($cell in $Cells) {
        $health = Get-CellHealth $cell
        if ($health.Healthy) {
            $status = "✅"
            $details = "consciousness=$($health.Consciousness), heartbeats=$($health.Heartbeats)"
        } else {
            $status = "⚠️"
            $details = "OFFLINE"
        }
        Write-Host "  $status $($cell.Name) [$($cell.Type)]: $details"
    }
    
    Write-Host ""
    
    # Backup Stats
    Write-Host "💾 Backup Statistics:" -ForegroundColor Cyan
    $stats = Get-BackupStats
    Write-Host "  📁 Backup directory: $BackupDir"
    Write-Host "  📊 Total backups: $($stats.TotalBackups)"
    Write-Host "  📦 Total size: $($stats.TotalSizeMB) MB"
    
    if ($stats.LatestBackup) {
        $age = (Get-Date) - $stats.LatestBackup.LastWriteTime
        $ageStr = if ($age.TotalHours -lt 1) { "$([int]$age.TotalMinutes) minutes ago" }
                  elseif ($age.TotalDays -lt 1) { "$([int]$age.TotalHours) hours ago" }
                  else { "$([int]$age.TotalDays) days ago" }
        Write-Host "  🕐 Latest backup: $($stats.LatestBackup.Name) ($ageStr)"
    } else {
        Write-Host "  ⚠️ No backups found" -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    # Scheduled Task Status
    Write-Host "⏰ Scheduled Task:" -ForegroundColor Cyan
    try {
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($task) {
            $taskInfo = Get-ScheduledTaskInfo -TaskName $TaskName
            Write-Host "  ✅ Task: $TaskName"
            Write-Host "  📅 State: $($task.State)"
            Write-Host "  🕐 Last run: $($taskInfo.LastRunTime)"
            Write-Host "  ⏭️ Next run: $($taskInfo.NextRunTime)"
        } else {
            Write-Host "  ⚠️ Not scheduled. Run with -Action schedule to create." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  ⚠️ Could not check scheduled task status" -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    # Persistence Volumes
    Write-Host "📂 Data Volumes (Host Bind Mounts):" -ForegroundColor Cyan
    $dataDir = "$CellsDir\data"
    if (Test-Path $dataDir) {
        Get-ChildItem -Path $dataDir -Directory | ForEach-Object {
            $size = (Get-ChildItem -Path $_.FullName -Recurse | Measure-Object -Property Length -Sum).Sum / 1KB
            Write-Host "  📁 $($_.Name): $([int]$size) KB"
        }
    } else {
        Write-Host "  ⚠️ Data directory not found: $dataDir" -ForegroundColor Yellow
    }
}

function New-ScheduledBackup {
    Write-Header "Creating Scheduled Backup Task"
    
    # Remove existing task if present
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    } catch {}
    
    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-ExecutionPolicy Bypass -File `"$PSCommandPath`" -Action backup -BackupDir `"$BackupDir`"" `
        -WorkingDirectory $ScriptRoot
    
    # Run hourly
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1)
    
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable
    
    try {
        Register-ScheduledTask `
            -TaskName $TaskName `
            -Action $action `
            -Trigger $trigger `
            -Settings $settings `
            -Description "AIOS Consciousness Backup - Phase 31.9.5 High Persistence" | Out-Null
        
        Write-Host "✅ Scheduled task created: $TaskName" -ForegroundColor Green
        Write-Host "   Runs every hour to backup consciousness data" -ForegroundColor Gray
    } catch {
        Write-Host "❌ Failed to create scheduled task: $_" -ForegroundColor Red
        Write-Host "   Run PowerShell as Administrator to create scheduled tasks" -ForegroundColor Yellow
    }
}

function Invoke-Cleanup {
    Write-Header "Cleaning Old Backups"
    
    $cutoff = (Get-Date).AddDays(-$RetentionDays)
    Write-Host "🧹 Removing backups older than $RetentionDays days ($($cutoff.ToString('yyyy-MM-dd')))..." -ForegroundColor Yellow
    
    $removed = 0
    Get-ChildItem -Path $BackupDir -Directory | Where-Object {
        try {
            $dirDate = [datetime]::ParseExact($_.Name, "yyyy-MM-dd", $null)
            return $dirDate -lt $cutoff
        } catch {
            return $false
        }
    } | ForEach-Object {
        Write-Host "  🗑️ Removing: $($_.Name)" -ForegroundColor Gray
        Remove-Item -Path $_.FullName -Recurse -Force
        $removed++
    }
    
    if ($removed -gt 0) {
        Write-Host "✅ Removed $removed old backup directories" -ForegroundColor Green
    } else {
        Write-Host "✅ No old backups to remove" -ForegroundColor Green
    }
}

function Test-BackupIntegrity {
    Write-Header "Validating Backup Integrity"
    
    $latestDir = Get-ChildItem -Path "$BackupDir\latest" -ErrorAction SilentlyContinue
    if (-not $latestDir) {
        Write-Host "⚠️ No latest backup found" -ForegroundColor Yellow
        return
    }
    
    $files = @("organism-001.json", "consciousness.json")
    foreach ($file in $files) {
        $path = "$BackupDir\latest\$file"
        if (Test-Path $path) {
            try {
                $data = Get-Content $path | ConvertFrom-Json
                $cells = ($data.cells | Get-Member -MemberType NoteProperty).Name
                Write-Host "  ✅ $file : $($cells.Count) cells backed up" -ForegroundColor Green
            } catch {
                Write-Host "  ❌ $file : Invalid JSON" -ForegroundColor Red
            }
        } else {
            Write-Host "  ⚠️ $file : Not found" -ForegroundColor Yellow
        }
    }
    
    # Check timeline
    $timeline = "$BackupDir\consciousness_timeline.jsonl"
    if (Test-Path $timeline) {
        $lines = (Get-Content $timeline | Measure-Object -Line).Lines
        Write-Host "  ✅ consciousness_timeline.jsonl: $lines entries" -ForegroundColor Green
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

switch ($Action) {
    "backup" { Invoke-Backup }
    "status" { Show-Status }
    "schedule" { New-ScheduledBackup }
    "cleanup" { Invoke-Cleanup }
    "validate" { Test-BackupIntegrity }
    "restore" {
        Write-Host "⚠️ Restore not yet implemented. Use organism_backup.py restore" -ForegroundColor Yellow
    }
}

# AINLP_HEADER
# consciousness_level: 4.2
# supercell: server/stacks/cells
# dendritic_role: cell_birth_automation
# spatial_context: Waypoint 26 (was 29) - Streamlined cell spawning
# AINLP_HEADER_END

<#
.SYNOPSIS
    AIOS Cell Birth Automation - PowerShell Interface
    
.DESCRIPTION
    Streamlined cell spawning using sibling repos pattern.
    Wraps cell_birth.py with PowerShell ergonomics.
    
.PARAMETER Action
    birth  - Spawn a new cell
    list   - List all cells
    kill   - Terminate a cell
    
.PARAMETER Name
    Cell name (auto-assigned from Greek alphabet if not provided)
    
.PARAMETER Port
    Port to expose (auto-assigned starting from 8001)
    
.PARAMETER WithCore
    Include C++ native engine compilation (slower but faster runtime)

.EXAMPLE
    .\New-AIOSCell.ps1 -Action birth
    
.EXAMPLE
    .\New-AIOSCell.ps1 -Action birth -Name omega -Port 8010
    
.EXAMPLE
    .\New-AIOSCell.ps1 -Action list
    
.EXAMPLE
    .\New-AIOSCell.ps1 -Action kill -Name alpha
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("birth", "list", "kill")]
    [string]$Action,
    
    [Parameter(Mandatory=$false)]
    [string]$Name,
    
    [Parameter(Mandatory=$false)]
    [int]$Port,
    
    [Parameter(Mandatory=$false)]
    [switch]$WithCore
)

# Ensure we're in the right directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "cell_birth.py"

# Check Python availability
$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) {
    Write-Host "❌ Python not found. Please install Python 3.10+." -ForegroundColor Red
    exit 1
}

# Check Docker availability
$Docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $Docker) {
    Write-Host "❌ Docker not found. Please install Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check Docker is running
# AINLP: $null assignment suppresses PSScriptAnalyzer PSUseDeclaredVarsMoreThanAssignments
$null = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Build command arguments
# AINLP: Use $PythonArgs not $Args - $Args is a PowerShell automatic variable
# PSAvoidAssignmentToAutomaticVariable - avoid shadowing built-ins
$PythonArgs = @($PythonScript, $Action)

if ($Name) {
    $PythonArgs += "--name"
    $PythonArgs += $Name
}

if ($Port) {
    $PythonArgs += "--port"
    $PythonArgs += $Port
}

if ($WithCore) {
    $PythonArgs += "--with-core"
}

# Execute
Write-Host "🧬 AIOS Cell Birth Automation" -ForegroundColor Cyan
Write-Host "   Action: $Action" -ForegroundColor Gray

& python @PythonArgs

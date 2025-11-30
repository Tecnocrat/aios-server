#!/usr/bin/env pwsh
<#
.SYNOPSIS
    AIOS Observability Stack Health Testing Workbench
    
.DESCRIPTION
    Comprehensive testing suite for Grafana, Traefik, and Prometheus
    Validates connectivity, health endpoints, metrics collection, and integration
    
.NOTES
    Author: AIOS Principal Software Architect Agent
    Date: 2025-11-22
    Consciousness Level: 3.26
#>

param(
    [switch]$Detailed,
    [switch]$OpenBrowsers,
    [switch]$ExportReport
)

# Color functions
function Write-Success { param($msg) Write-Host "✓ $msg" -ForegroundColor Green }
function Write-Error { param($msg) Write-Host "✗ $msg" -ForegroundColor Red }
function Write-Warning { param($msg) Write-Host "⚠ $msg" -ForegroundColor Yellow }
function Write-Info { param($msg) Write-Host "ℹ $msg" -ForegroundColor Cyan }
function Write-Section { param($msg) Write-Host "`n═══ $msg ═══`n" -ForegroundColor Magenta }

# Test results collector
$global:TestResults = @{
    Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    TotalTests = 0
    Passed = 0
    Failed = 0
    Warnings = 0
    Tests = @()
}

function Add-TestResult {
    param(
        [string]$Category,
        [string]$Test,
        [string]$Status,  # Pass, Fail, Warning
        [string]$Details,
        [object]$Data = $null
    )
    
    $global:TestResults.TotalTests++
    switch ($Status) {
        "Pass" { $global:TestResults.Passed++ }
        "Fail" { $global:TestResults.Failed++ }
        "Warning" { $global:TestResults.Warnings++ }
    }
    
    $global:TestResults.Tests += @{
        Category = $Category
        Test = $Test
        Status = $Status
        Details = $Details
        Data = $Data
    }
}

# ═══════════════════════════════════════════════════════════════
# Docker Infrastructure Tests
# ═══════════════════════════════════════════════════════════════

Write-Section "Docker Infrastructure Validation"

# Test 1: Docker daemon
try {
    docker version | Out-Null
    Write-Success "Docker daemon: Running"
    Add-TestResult -Category "Infrastructure" -Test "Docker Daemon" -Status "Pass" -Details "Docker daemon accessible"
} catch {
    Write-Error "Docker daemon: Not running"
    Add-TestResult -Category "Infrastructure" -Test "Docker Daemon" -Status "Fail" -Details $_.Exception.Message
    exit 1
}

# Test 2: Container status
Write-Info "Checking AIOS containers..."
$containers = docker ps --filter "name=aios-" --format "{{.Names}}|{{.Status}}" | ForEach-Object {
    $parts = $_ -split '\|'
    @{ Name = $parts[0]; Status = $parts[1] }
}

$expectedContainers = @('aios-prometheus', 'aios-grafana', 'aios-traefik')
foreach ($expected in $expectedContainers) {
    $container = $containers | Where-Object { $_.Name -eq $expected }
    if ($container) {
        if ($container.Status -match "Up") {
            Write-Success "${expected}: $($container.Status)"
            Add-TestResult -Category "Infrastructure" -Test "Container $expected" -Status "Pass" -Details $container.Status
        } else {
            Write-Warning "${expected}: $($container.Status)"
            Add-TestResult -Category "Infrastructure" -Test "Container $expected" -Status "Warning" -Details $container.Status
        }
    } else {
        Write-Error "${expected}: Not found"
        Add-TestResult -Category "Infrastructure" -Test "Container $expected" -Status "Fail" -Details "Container not running"
    }
}

# Test 3: Networks
Write-Info "Checking Docker networks..."
$networks = docker network ls --filter "name=aios-" --format "{{.Name}}"
$expectedNetworks = @('aios-observability', 'aios-ingress')
foreach ($expected in $expectedNetworks) {
    if ($networks -contains $expected) {
        Write-Success "Network ${expected}: Exists"
        Add-TestResult -Category "Infrastructure" -Test "Network $expected" -Status "Pass" -Details "Network configured"
    } else {
        Write-Error "Network ${expected}: Missing"
        Add-TestResult -Category "Infrastructure" -Test "Network $expected" -Status "Fail" -Details "Network not found"
    }
}

# ═══════════════════════════════════════════════════════════════
# Prometheus Tests
# ═══════════════════════════════════════════════════════════════

Write-Section "Prometheus Health & Metrics"

# Test 4: Prometheus port accessibility
try {
    $prometheusPort = Test-NetConnection -ComputerName localhost -Port 9090 -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($prometheusPort) {
        Write-Success "Prometheus port 9090: Accessible"
        Add-TestResult -Category "Prometheus" -Test "Port Accessibility" -Status "Pass" -Details "Port 9090 responding"
    } else {
        throw "Port not accessible"
    }
} catch {
    Write-Error "Prometheus port 9090: Not accessible"
    Add-TestResult -Category "Prometheus" -Test "Port Accessibility" -Status "Fail" -Details $_.Exception.Message
}

# Test 5: Prometheus health endpoint
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Success "Prometheus health: $($response.Content)"
        Add-TestResult -Category "Prometheus" -Test "Health Endpoint" -Status "Pass" -Details $response.Content -Data $response
    }
} catch {
    Write-Error "Prometheus health: Failed - $($_.Exception.Message)"
    Add-TestResult -Category "Prometheus" -Test "Health Endpoint" -Status "Fail" -Details $_.Exception.Message
}

# Test 6: Prometheus ready endpoint
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9090/-/ready" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Success "Prometheus ready: $($response.Content)"
        Add-TestResult -Category "Prometheus" -Test "Ready Endpoint" -Status "Pass" -Details $response.Content
    }
} catch {
    Write-Error "Prometheus ready: Not ready - $($_.Exception.Message)"
    Add-TestResult -Category "Prometheus" -Test "Ready Endpoint" -Status "Fail" -Details $_.Exception.Message
}

# Test 7: Prometheus targets
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/targets" -UseBasicParsing -TimeoutSec 5
    $targets = ($response.Content | ConvertFrom-Json).data.activeTargets
    
    Write-Info "Prometheus targets: $($targets.Count) active"
    
    $upTargets = ($targets | Where-Object { $_.health -eq "up" }).Count
    $downTargets = ($targets | Where-Object { $_.health -eq "down" }).Count
    
    if ($Detailed) {
        foreach ($target in $targets) {
            $health = if ($target.health -eq "up") { "✓" } else { "✗" }
            Write-Host "  $health $($target.labels.job): $($target.health)" -ForegroundColor $(if ($target.health -eq "up") { "Green" } else { "Red" })
        }
    }
    
    Write-Success "Targets UP: $upTargets"
    if ($downTargets -gt 0) {
        Write-Warning "Targets DOWN: $downTargets"
    }
    
    Add-TestResult -Category "Prometheus" -Test "Targets Status" -Status $(if ($downTargets -eq 0) { "Pass" } else { "Warning" }) `
        -Details "UP: $upTargets, DOWN: $downTargets" -Data $targets
        
} catch {
    Write-Error "Prometheus targets: Failed to query - $($_.Exception.Message)"
    Add-TestResult -Category "Prometheus" -Test "Targets Status" -Status "Fail" -Details $_.Exception.Message
}

# Test 8: Prometheus metrics query
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=up" -UseBasicParsing -TimeoutSec 5
    $data = ($response.Content | ConvertFrom-Json).data.result
    
    Write-Success "Prometheus query: Returned $($data.Count) metrics"
    Add-TestResult -Category "Prometheus" -Test "Metrics Query" -Status "Pass" -Details "$($data.Count) metrics available"
    
} catch {
    Write-Error "Prometheus query: Failed - $($_.Exception.Message)"
    Add-TestResult -Category "Prometheus" -Test "Metrics Query" -Status "Fail" -Details $_.Exception.Message
}

# Test 9: Prometheus configuration
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/status/config" -UseBasicParsing -TimeoutSec 5
    $config = ($response.Content | ConvertFrom-Json).data.yaml
    
    Write-Success "Prometheus config: Loaded"
    
    if ($Detailed) {
        $scrapeConfigs = ($config | Select-String -Pattern "job_name:" -AllMatches).Matches.Count
        Write-Info "  Scrape jobs configured: $scrapeConfigs"
    }
    
    Add-TestResult -Category "Prometheus" -Test "Configuration" -Status "Pass" -Details "Config loaded successfully"
    
} catch {
    Write-Error "Prometheus config: Failed to load - $($_.Exception.Message)"
    Add-TestResult -Category "Prometheus" -Test "Configuration" -Status "Fail" -Details $_.Exception.Message
}

# ═══════════════════════════════════════════════════════════════
# Grafana Tests
# ═══════════════════════════════════════════════════════════════

Write-Section "Grafana Dashboard & Datasources"

# Test 10: Grafana port accessibility
try {
    $grafanaPort = Test-NetConnection -ComputerName localhost -Port 3000 -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($grafanaPort) {
        Write-Success "Grafana port 3000: Accessible"
        Add-TestResult -Category "Grafana" -Test "Port Accessibility" -Status "Pass" -Details "Port 3000 responding"
    } else {
        throw "Port not accessible"
    }
} catch {
    Write-Error "Grafana port 3000: Not accessible"
    Add-TestResult -Category "Grafana" -Test "Port Accessibility" -Status "Fail" -Details $_.Exception.Message
}

# Test 11: Grafana health endpoint
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -UseBasicParsing -TimeoutSec 5
    $health = $response.Content | ConvertFrom-Json
    
    Write-Success "Grafana health: Database=$($health.database), Version=$($health.version)"
    Add-TestResult -Category "Grafana" -Test "Health Endpoint" -Status "Pass" `
        -Details "Database: $($health.database), Version: $($health.version)" -Data $health
        
} catch {
    Write-Error "Grafana health: Failed - $($_.Exception.Message)"
    Add-TestResult -Category "Grafana" -Test "Health Endpoint" -Status "Fail" -Details $_.Exception.Message
}

# Test 12: Grafana login page
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000/login" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Success "Grafana login: Accessible"
        Add-TestResult -Category "Grafana" -Test "Login Page" -Status "Pass" -Details "Login page loading"
    }
} catch {
    Write-Error "Grafana login: Failed - $($_.Exception.Message)"
    Add-TestResult -Category "Grafana" -Test "Login Page" -Status "Fail" -Details $_.Exception.Message
}

# Test 13: Grafana datasources (requires auth)
try {
    $cred = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("aios:6996"))
    $headers = @{ Authorization = "Basic $cred" }
    
    $response = Invoke-WebRequest -Uri "http://localhost:3000/api/datasources" -Headers $headers -UseBasicParsing -TimeoutSec 5
    $datasources = $response.Content | ConvertFrom-Json
    
    Write-Success "Grafana datasources: $($datasources.Count) configured"
    
    if ($Detailed) {
        foreach ($ds in $datasources) {
            Write-Info "  - $($ds.name) ($($ds.type)): $($ds.url)"
        }
    }
    
    Add-TestResult -Category "Grafana" -Test "Datasources" -Status "Pass" -Details "$($datasources.Count) datasources" -Data $datasources
    
} catch {
    Write-Warning "Grafana datasources: Could not query (auth required)"
    Add-TestResult -Category "Grafana" -Test "Datasources" -Status "Warning" -Details "Authentication required or not configured"
}

# ═══════════════════════════════════════════════════════════════
# Traefik Tests
# ═══════════════════════════════════════════════════════════════

Write-Section "Traefik Ingress & Routing"

# Test 14: Traefik dashboard port
try {
    $traefikPort = Test-NetConnection -ComputerName localhost -Port 8080 -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($traefikPort) {
        Write-Success "Traefik dashboard port 8080: Accessible"
        Add-TestResult -Category "Traefik" -Test "Dashboard Port" -Status "Pass" -Details "Port 8080 responding"
    } else {
        throw "Port not accessible"
    }
} catch {
    Write-Error "Traefik dashboard port 8080: Not accessible"
    Add-TestResult -Category "Traefik" -Test "Dashboard Port" -Status "Fail" -Details $_.Exception.Message
}

# Test 15: Traefik HTTP entrypoint
try {
    $httpPort = Test-NetConnection -ComputerName localhost -Port 80 -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($httpPort) {
        Write-Success "Traefik HTTP port 80: Accessible"
        Add-TestResult -Category "Traefik" -Test "HTTP Entrypoint" -Status "Pass" -Details "Port 80 responding"
    } else {
        throw "Port not accessible"
    }
} catch {
    Write-Error "Traefik HTTP port 80: Not accessible"
    Add-TestResult -Category "Traefik" -Test "HTTP Entrypoint" -Status "Fail" -Details $_.Exception.Message
}

# Test 16: Traefik HTTPS entrypoint
try {
    $httpsPort = Test-NetConnection -ComputerName localhost -Port 443 -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($httpsPort) {
        Write-Success "Traefik HTTPS port 443: Accessible"
        Add-TestResult -Category "Traefik" -Test "HTTPS Entrypoint" -Status "Pass" -Details "Port 443 responding"
    } else {
        throw "Port not accessible"
    }
} catch {
    Write-Error "Traefik HTTPS port 443: Not accessible"
    Add-TestResult -Category "Traefik" -Test "HTTPS Entrypoint" -Status "Fail" -Details $_.Exception.Message
}

# Test 17: Traefik metrics port
try {
    $metricsPort = Test-NetConnection -ComputerName localhost -Port 8082 -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($metricsPort) {
        Write-Success "Traefik metrics port 8082: Accessible"
        Add-TestResult -Category "Traefik" -Test "Metrics Port" -Status "Pass" -Details "Port 8082 responding"
    } else {
        throw "Port not accessible"
    }
} catch {
    Write-Warning "Traefik metrics port 8082: Not accessible"
    Add-TestResult -Category "Traefik" -Test "Metrics Port" -Status "Warning" -Details $_.Exception.Message
}

# Test 18: Traefik dashboard accessibility
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8080/dashboard/" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Success "Traefik dashboard: Accessible"
        Add-TestResult -Category "Traefik" -Test "Dashboard UI" -Status "Pass" -Details "Dashboard loading"
    }
} catch {
    Write-Error "Traefik dashboard: Failed - $($_.Exception.Message)"
    Add-TestResult -Category "Traefik" -Test "Dashboard UI" -Status "Fail" -Details $_.Exception.Message
}

# Test 19: Traefik health check
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8080/ping" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Success "Traefik health: $($response.Content)"
        Add-TestResult -Category "Traefik" -Test "Health Check" -Status "Pass" -Details $response.Content
    }
} catch {
    Write-Error "Traefik health: Failed - $($_.Exception.Message)"
    Add-TestResult -Category "Traefik" -Test "Health Check" -Status "Fail" -Details $_.Exception.Message
}

# ═══════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════

Write-Section "Service Integration Tests"

# Test 20: Prometheus scraping Traefik
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=traefik_entrypoint_requests_total" -UseBasicParsing -TimeoutSec 5
    $data = ($response.Content | ConvertFrom-Json).data.result
    
    if ($data.Count -gt 0) {
        Write-Success "Prometheus → Traefik: Scraping metrics ($($data.Count) series)"
        Add-TestResult -Category "Integration" -Test "Prometheus scrapes Traefik" -Status "Pass" -Details "$($data.Count) metric series"
    } else {
        Write-Warning "Prometheus → Traefik: No metrics yet (may need time to scrape)"
        Add-TestResult -Category "Integration" -Test "Prometheus scrapes Traefik" -Status "Warning" -Details "No metrics found yet"
    }
} catch {
    Write-Error "Prometheus → Traefik: Integration failed - $($_.Exception.Message)"
    Add-TestResult -Category "Integration" -Test "Prometheus scrapes Traefik" -Status "Fail" -Details $_.Exception.Message
}

# Test 21: Prometheus scraping cAdvisor
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=container_cpu_usage_seconds_total" -UseBasicParsing -TimeoutSec 5
    $data = ($response.Content | ConvertFrom-Json).data.result
    
    if ($data.Count -gt 0) {
        Write-Success "Prometheus → cAdvisor: Scraping container metrics ($($data.Count) series)"
        Add-TestResult -Category "Integration" -Test "Prometheus scrapes cAdvisor" -Status "Pass" -Details "$($data.Count) metric series"
    } else {
        Write-Warning "Prometheus → cAdvisor: No metrics yet"
        Add-TestResult -Category "Integration" -Test "Prometheus scrapes cAdvisor" -Status "Warning" -Details "No metrics found yet"
    }
} catch {
    Write-Error "Prometheus → cAdvisor: Integration failed - $($_.Exception.Message)"
    Add-TestResult -Category "Integration" -Test "Prometheus scrapes cAdvisor" -Status "Fail" -Details $_.Exception.Message
}

# Test 22: Loki accessibility for Grafana
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3100/ready" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Success "Loki: Ready for Grafana logs"
        Add-TestResult -Category "Integration" -Test "Loki Ready" -Status "Pass" -Details "Loki accepting log queries"
    }
} catch {
    Write-Error "Loki: Not accessible - $($_.Exception.Message)"
    Add-TestResult -Category "Integration" -Test "Loki Ready" -Status "Fail" -Details $_.Exception.Message
}

# ═══════════════════════════════════════════════════════════════
# Results Summary
# ═══════════════════════════════════════════════════════════════

Write-Section "Test Results Summary"

$passRate = [math]::Round(($global:TestResults.Passed / $global:TestResults.TotalTests) * 100, 2)

Write-Host "Total Tests: $($global:TestResults.TotalTests)" -ForegroundColor White
Write-Success "Passed: $($global:TestResults.Passed)"
if ($global:TestResults.Failed -gt 0) {
    Write-Error "Failed: $($global:TestResults.Failed)"
}
if ($global:TestResults.Warnings -gt 0) {
    Write-Warning "Warnings: $($global:TestResults.Warnings)"
}
Write-Host "Pass Rate: $passRate%" -ForegroundColor $(if ($passRate -ge 90) { "Green" } elseif ($passRate -ge 70) { "Yellow" } else { "Red" })

# Export report if requested
if ($ExportReport) {
    # AINLP.dendritic: Reports stored in sibling reports/ directory
    $reportsDir = Join-Path $PSScriptRoot "..\reports"
    if (-not (Test-Path $reportsDir)) { New-Item -ItemType Directory -Path $reportsDir -Force | Out-Null }
    $reportPath = Join-Path $reportsDir "test_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    $global:TestResults | ConvertTo-Json -Depth 10 | Out-File -FilePath $reportPath -Encoding utf8
    Write-Info "Report exported to: $reportPath"
}

# Open browsers if requested
if ($OpenBrowsers) {
    Write-Info "`nOpening service dashboards..."
    Start-Process "http://localhost:9090"  # Prometheus
    Start-Sleep -Milliseconds 500
    Start-Process "http://localhost:3000"  # Grafana
    Start-Sleep -Milliseconds 500
    Start-Process "http://localhost:8080/dashboard/"  # Traefik
}

# Return exit code based on failures
if ($global:TestResults.Failed -gt 0) {
    exit 1
} else {
    exit 0
}

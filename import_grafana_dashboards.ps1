#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Import community dashboards into Grafana
    
.DESCRIPTION
    Automatically imports recommended dashboards for Docker, Traefik, cAdvisor, and Node Exporter
    
.NOTES
    Grafana API: http://localhost:3000
    Credentials: aios / 6996
#>

$grafanaUrl = "http://localhost:3000"
$cred = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("aios:6996"))
$headers = @{
    Authorization = "Basic $cred"
    "Content-Type" = "application/json"
}

# Dashboard IDs to import
$dashboards = @(
    @{
        Id = 893
        Name = "Docker & System Monitoring"
        Description = "Comprehensive Docker container and system metrics"
    },
    @{
        Id = 4475
        Name = "Traefik 2.0 Dashboard"
        Description = "Traefik ingress routing and traffic metrics"
    },
    @{
        Id = 14282
        Name = "cAdvisor exporter"
        Description = "Container resource usage and performance"
    },
    @{
        Id = 1860
        Name = "Node Exporter Full"
        Description = "Complete host system metrics (CPU, memory, disk, network)"
    }
)

Write-Host "`n═══ Grafana Dashboard Import ═══`n" -ForegroundColor Magenta

# Get Prometheus datasource UID
try {
    $datasources = Invoke-RestMethod -Uri "$grafanaUrl/api/datasources" -Headers $headers -Method Get
    $promDs = $datasources | Where-Object { $_.type -eq "prometheus" } | Select-Object -First 1
    
    if (-not $promDs) {
        Write-Host "✗ Prometheus datasource not found" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✓ Found Prometheus datasource: $($promDs.name) (UID: $($promDs.uid))" -ForegroundColor Green
    
} catch {
    Write-Host "✗ Failed to get datasources: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Import each dashboard
foreach ($dash in $dashboards) {
    Write-Host "`nImporting: $($dash.Name) (ID: $($dash.Id))..." -ForegroundColor Cyan
    
    try {
        # Grafana dashboard import payload
        $body = @{
            dashboard = @{
                id = $null
                uid = $null
                title = $dash.Name
            }
            folderId = 0
            overwrite = $true
            inputs = @(
                @{
                    name = "DS_PROMETHEUS"
                    type = "datasource"
                    pluginId = "prometheus"
                    value = $promDs.uid
                }
            )
            pluginId = ""
            gnetId = $dash.Id
        } | ConvertTo-Json -Depth 10
        
        $response = Invoke-RestMethod -Uri "$grafanaUrl/api/dashboards/import" -Headers $headers -Method Post -Body $body
        
        Write-Host "  ✓ Imported: $($response.importedTitle)" -ForegroundColor Green
        Write-Host "  📊 URL: $grafanaUrl/d/$($response.importedUid)" -ForegroundColor Gray
        
    } catch {
        $errorMsg = $_.ErrorDetails.Message | ConvertFrom-Json -ErrorAction SilentlyContinue
        if ($errorMsg.message -match "already exists") {
            Write-Host "  ⚠ Dashboard already exists, updating..." -ForegroundColor Yellow
        } else {
            Write-Host "  ✗ Failed: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

Write-Host "`n✓ Dashboard import complete" -ForegroundColor Green
Write-Host "Access Grafana: $grafanaUrl (aios / 6996)`n" -ForegroundColor Cyan

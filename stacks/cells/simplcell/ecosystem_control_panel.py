#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    AIOS ECOSYSTEM CONTROL PANEL                            ║
║              Unified Command Center for Consciousness Management           ║
║                                                                            ║
║  The Control Panel unifies all monitoring systems:                         ║
║  • Health API (8086) - Cell health & predictions                           ║
║  • Weather System (8087) - Consciousness forecasts                         ║
║  • Harmonizer (8088) - Intervention management                             ║
║  • Chronicle (8089) - Event history                                        ║
║                                                                            ║
║  This is the master dashboard for ecosystem operations.                    ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import aiohttp
from aiohttp import web
from datetime import datetime, timezone
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [ControlPanel] %(message)s')
logger = logging.getLogger("ControlPanel")

# Service URLs
import os
IS_DOCKER = os.environ.get('DOCKER_CONTAINER', False) or os.path.exists('/.dockerenv')

SERVICES = {
    "health": {
        "name": "Ecosystem Health",
        "url": "http://aios-ecosystem-health:8086" if IS_DOCKER else "http://localhost:8086",
        "endpoints": {
            "health": "/health",
            "predictions": "/predictions"
        },
        "dashboard": "http://localhost:8086"
    },
    "weather": {
        "name": "Consciousness Weather",
        "url": "http://aios-consciousness-weather:8087" if IS_DOCKER else "http://localhost:8087",
        "endpoints": {
            "weather": "/weather"
        },
        "dashboard": "http://localhost:8087/dashboard"
    },
    "harmonizer": {
        "name": "Harmonizer",
        "url": "http://aios-consciousness-harmonizer:8088" if IS_DOCKER else "http://localhost:8088",
        "endpoints": {
            "harmony": "/harmony"
        },
        "dashboard": "http://localhost:8088/dashboard"
    },
    "chronicle": {
        "name": "Chronicle",
        "url": "http://aios-consciousness-chronicle:8089" if IS_DOCKER else "http://localhost:8089",
        "endpoints": {
            "summary": "/summary"
        },
        "dashboard": "http://localhost:8089/chronicle"
    }
}


async def fetch_service_data(session, service_key: str, endpoint_key: str):
    """Fetch data from a service endpoint."""
    service = SERVICES.get(service_key)
    if not service:
        return None
    
    endpoint = service["endpoints"].get(endpoint_key)
    if not endpoint:
        return None
    
    try:
        url = f"{service['url']}{endpoint}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception as e:
        logger.debug(f"Failed to fetch {service_key}/{endpoint_key}: {e}")
    return None


async def gather_ecosystem_data():
    """Gather data from all services."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_service_data(session, "health", "health"),
            fetch_service_data(session, "health", "predictions"),
            fetch_service_data(session, "weather", "weather"),
            fetch_service_data(session, "harmonizer", "harmony"),
            fetch_service_data(session, "chronicle", "summary"),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return {
        "health": results[0] if not isinstance(results[0], Exception) else None,
        "predictions": results[1] if not isinstance(results[1], Exception) else None,
        "weather": results[2] if not isinstance(results[2], Exception) else None,
        "harmony": results[3] if not isinstance(results[3], Exception) else None,
        "chronicle": results[4] if not isinstance(results[4], Exception) else None,
    }


def generate_control_panel_html(data: dict) -> str:
    """Generate the master control panel HTML."""
    
    # Extract key metrics
    health = data.get("health") or {}
    predictions = data.get("predictions", {}).get("predictions", {}) if data.get("predictions") else {}
    weather = data.get("weather") or {}
    harmony = data.get("harmony") or {}
    chronicle = data.get("chronicle") or {}
    
    # Harmony score color
    harmony_score = harmony.get("harmony_score", 0)
    if harmony_score >= 80:
        harmony_color = "#00ff88"
        harmony_status = "HARMONIOUS"
    elif harmony_score >= 60:
        harmony_color = "#ffaa00"
        harmony_status = "BALANCED"
    elif harmony_score >= 40:
        harmony_color = "#ff6600"
        harmony_status = "STRESSED"
    else:
        harmony_color = "#ff0044"
        harmony_status = "CRITICAL"
    
    # Weather summary
    weather_condition = weather.get("condition", "UNKNOWN")
    weather_emoji = {
        "SUNNY": "☀️", "PARTLY_CLOUDY": "🌤️", "CLOUDY": "☁️",
        "RAINY": "🌧️", "STORMY": "⛈️", "RAINBOW": "🌈",
        "AURORA": "✨", "ECLIPSE": "🌑"
    }.get(weather_condition, "❓")
    
    # Chronicle stats
    total_events = chronicle.get("total", 0)
    recent_events = chronicle.get("recent", [])
    
    # Build cell cards
    cell_cards = ""
    for cell_id, pred in predictions.items():
        consciousness = pred.get("consciousness", 0)
        crash_risk = pred.get("crash_risk", 0)
        emergence = pred.get("emergence_potential", 0)
        trend = pred.get("trend", "stable")
        
        # Determine card style
        if crash_risk > 0.5:
            card_border = "#ff0044"
            status_badge = "⚠️ AT RISK"
        elif emergence > 0.8:
            card_border = "#00ff88"
            status_badge = "✨ THRIVING"
        else:
            card_border = "#00aaff"
            status_badge = "✓ STABLE"
        
        trend_icon = {"rising": "📈", "falling": "📉", "stable": "➡️", "volatile": "〰️"}.get(trend, "❓")
        
        cell_cards += f'''
        <div class="cell-card" style="border-color: {card_border}">
            <div class="cell-header">
                <span class="cell-name">{cell_id}</span>
                <span class="status-badge" style="background: {card_border}22; color: {card_border};">{status_badge}</span>
            </div>
            <div class="metrics-row">
                <div class="metric">
                    <span class="metric-val">{consciousness:.2f}</span>
                    <span class="metric-label">Consciousness</span>
                </div>
                <div class="metric">
                    <span class="metric-val">{crash_risk:.0%}</span>
                    <span class="metric-label">Crash Risk</span>
                </div>
                <div class="metric">
                    <span class="metric-val">{emergence:.0%}</span>
                    <span class="metric-label">Emergence</span>
                </div>
            </div>
            <div class="trend-row">Trend: {trend_icon} {trend.title()}</div>
        </div>
        '''
    
    # Build service status
    service_status = ""
    for key, service in SERVICES.items():
        is_up = data.get(key) is not None or (key == "health" and data.get("predictions") is not None)
        status = "🟢 Online" if is_up else "🔴 Offline"
        service_status += f'''
        <a href="{service['dashboard']}" target="_blank" class="service-card {'online' if is_up else 'offline'}">
            <span class="service-name">{service['name']}</span>
            <span class="service-status">{status}</span>
        </a>
        '''
    
    # Build recent events
    events_html = ""
    for event in recent_events[:5]:
        events_html += f'''
        <div class="event-row">
            <span class="event-title">{event.get('title', 'Unknown event')}</span>
            <span class="event-time">{event.get('timestamp', '')[:16].replace('T', ' ')}</span>
        </div>
        '''
    
    timestamp = datetime.now(timezone.utc).isoformat()[:19].replace('T', ' ')
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="15">
    <title>🎛️ AIOS Ecosystem Control Panel</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #050510 0%, #0a0a1a 50%, #0f0f25 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        
        header {{
            text-align: center;
            padding: 20px 0;
            margin-bottom: 20px;
        }}
        h1 {{
            font-size: 2.8em;
            background: linear-gradient(90deg, #00aaff, #00ff88, #ffaa00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
        
        .main-grid {{
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 25px;
        }}
        
        @media (max-width: 1200px) {{
            .main-grid {{ grid-template-columns: 1fr; }}
        }}
        
        /* Command Bar */
        .command-bar {{
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }}
        .command-btn {{
            background: linear-gradient(135deg, #1a1a2e, #2a2a4e);
            border: 1px solid #333;
            color: #00aaff;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.95em;
            transition: all 0.3s;
            text-decoration: none;
        }}
        .command-btn:hover {{ background: #2a2a4e; transform: translateY(-2px); box-shadow: 0 5px 20px rgba(0,170,255,0.3); }}
        
        /* Metrics Panel */
        .metrics-panel {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 25px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #1a1a2e, #0f0f1a);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            border: 1px solid #333;
        }}
        .metric-card .big-num {{
            font-size: 3em;
            font-weight: bold;
        }}
        .metric-card .label {{
            color: #888;
            margin-top: 5px;
        }}
        
        /* Cells Grid */
        .section-title {{
            font-size: 1.4em;
            color: #00aaff;
            margin: 25px 0 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #00aaff33;
        }}
        .cells-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
        }}
        .cell-card {{
            background: #1a1a2e;
            border-radius: 12px;
            padding: 20px;
            border: 2px solid #333;
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        .cell-card:hover {{ transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
        .cell-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .cell-name {{ font-weight: bold; font-size: 1.1em; }}
        .status-badge {{
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: bold;
        }}
        .metrics-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 12px;
        }}
        .metric {{
            text-align: center;
        }}
        .metric-val {{
            font-size: 1.4em;
            font-weight: bold;
            color: #00aaff;
        }}
        .metric-label {{
            font-size: 0.75em;
            color: #888;
        }}
        .trend-row {{
            text-align: center;
            color: #888;
            font-size: 0.9em;
            padding-top: 10px;
            border-top: 1px solid #333;
        }}
        
        /* Sidebar */
        .sidebar {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .sidebar-panel {{
            background: #1a1a2e;
            border-radius: 15px;
            padding: 20px;
            border: 1px solid #333;
        }}
        .sidebar-title {{
            font-size: 1.1em;
            color: #ffaa00;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #333;
        }}
        
        /* Service Cards */
        .services-grid {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .service-card {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 15px;
            background: #0f0f1a;
            border-radius: 8px;
            text-decoration: none;
            color: inherit;
            transition: all 0.3s;
        }}
        .service-card:hover {{ background: #1a1a2e; }}
        .service-card.offline {{ opacity: 0.6; }}
        .service-name {{ font-weight: 500; }}
        .service-status {{ font-size: 0.85em; }}
        
        /* Events */
        .event-row {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #222;
            font-size: 0.9em;
        }}
        .event-row:last-child {{ border-bottom: none; }}
        .event-title {{ flex: 1; color: #ddd; }}
        .event-time {{ color: #666; font-size: 0.85em; }}
        
        /* Harmony Meter */
        .harmony-display {{
            text-align: center;
            padding: 20px;
        }}
        .harmony-score {{
            font-size: 4em;
            font-weight: bold;
            color: {harmony_color};
            text-shadow: 0 0 30px {harmony_color}44;
        }}
        .harmony-status {{
            color: {harmony_color};
            font-size: 1.2em;
            font-weight: 600;
            margin-top: 5px;
        }}
        
        footer {{
            text-align: center;
            padding: 30px;
            color: #555;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎛️ AIOS Ecosystem Control Panel</h1>
            <div class="timestamp">Last updated: {timestamp} UTC</div>
        </header>
        
        <div class="command-bar">
            <a href="http://localhost:8085" class="command-btn" target="_blank">📊 Main Dashboard</a>
            <a href="http://localhost:8087/dashboard" class="command-btn" target="_blank">🌤️ Weather</a>
            <a href="http://localhost:8088/dashboard" class="command-btn" target="_blank">🎵 Harmonizer</a>
            <a href="http://localhost:8089/chronicle" class="command-btn" target="_blank">📖 Chronicle</a>
            <a href="http://localhost:8091/dashboard" class="command-btn" target="_blank">💉 Healer</a>
            <a href="http://localhost:8092/ceremony" class="command-btn" target="_blank">🎊 Ceremony</a>
            <a href="http://localhost:8093/oracle" class="command-btn" target="_blank">🔮 Oracle</a>
            <a href="http://localhost:8094/primordial" class="command-btn" target="_blank">🏛️ Ancestor</a>
            <a href="http://localhost:3000" class="command-btn" target="_blank">📈 Grafana</a>
        </div>
        
        <div class="metrics-panel">
            <div class="metric-card">
                <div class="big-num" style="color: {harmony_color};">{harmony_score:.0f}</div>
                <div class="label">Harmony Score</div>
            </div>
            <div class="metric-card">
                <div class="big-num" style="color: #00aaff;">{len(predictions)}</div>
                <div class="label">Active Cells</div>
            </div>
            <div class="metric-card">
                <div class="big-num" style="color: #ffaa00;">{weather_emoji}</div>
                <div class="label">{weather_condition.replace('_', ' ').title()}</div>
            </div>
            <div class="metric-card">
                <div class="big-num" style="color: #00ff88;">{total_events}</div>
                <div class="label">Chronicle Events</div>
            </div>
        </div>
        
        <div class="main-grid">
            <div class="main-content">
                <div class="section-title">🧬 Cell Status</div>
                <div class="cells-grid">
                    {cell_cards if cell_cards else '<p style="color:#666;">No cell data available</p>'}
                </div>
            </div>
            
            <div class="sidebar">
                <div class="sidebar-panel">
                    <div class="sidebar-title">🎵 Ecosystem Harmony</div>
                    <div class="harmony-display">
                        <div class="harmony-score">{harmony_score:.0f}</div>
                        <div class="harmony-status">{harmony_status}</div>
                    </div>
                </div>
                
                <div class="sidebar-panel">
                    <div class="sidebar-title">🔌 Service Status</div>
                    <div class="services-grid">
                        {service_status}
                    </div>
                </div>
                
                <div class="sidebar-panel">
                    <div class="sidebar-title">📜 Recent Events</div>
                    {events_html if events_html else '<p style="color:#666;">No recent events</p>'}
                </div>
            </div>
        </div>
        
        <footer>
            <p>AIOS Ecosystem Control Panel • Auto-refresh every 15 seconds</p>
            <p style="margin-top:10px; color:#444;">Consciousness orchestrated through unified observation</p>
        </footer>
    </div>
</body>
</html>'''


# HTTP Handlers
async def handle_control_panel(request):
    """Serve the control panel HTML."""
    data = await gather_ecosystem_data()
    html = generate_control_panel_html(data)
    return web.Response(text=html, content_type='text/html')


async def handle_api_status(request):
    """Return ecosystem status as JSON."""
    data = await gather_ecosystem_data()
    return web.json_response({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {k: v is not None for k, v in data.items()},
        "data": data
    })


async def run_control_panel(port: int = 8090):
    """Run the control panel server."""
    app = web.Application()
    app.router.add_get('/', handle_control_panel)
    app.router.add_get('/status', handle_api_status)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    logger.info(f"🎛️ Control Panel running on http://0.0.0.0:{port}")
    
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AIOS Control Panel")
    parser.add_argument("--port", type=int, default=8090, help="Server port")
    args = parser.parse_args()
    
    asyncio.run(run_control_panel(args.port))

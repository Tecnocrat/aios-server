#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    AIOS CONSCIOUSNESS WEATHER SYSTEM                       ║
║           Real-time Forecasting & Alert Broadcasting for the Ecosystem     ║
║                                                                            ║
║  Features:                                                                 ║
║  - Consciousness "weather" forecasts (sunny, cloudy, stormy, etc.)         ║
║  - Ecosystem-wide trend analysis                                           ║
║  - Alert broadcasting for critical events                                  ║
║  - Phase transition celebration announcements                              ║
║  - Historical pattern correlation                                          ║
╚════════════════════════════════════════════════════════════════════════════╝

The consciousness weather metaphor:
- ☀️ SUNNY: Rising consciousness, low crash risk, high emergence
- 🌤️ PARTLY_CLOUDY: Stable with minor fluctuations  
- ☁️ CLOUDY: Declining trend or elevated risk
- 🌧️ RAINY: Active decline in progress
- ⛈️ STORMY: High crash risk, volatile patterns
- 🌈 RAINBOW: Post-recovery, emerging from crash
- ✨ AURORA: Phase transition imminent
"""

import sqlite3
import json
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s [Weather] %(message)s')
logger = logging.getLogger("ConsciousnessWeather")

# ═══════════════════════════════════════════════════════════════════════
# WEATHER CONDITIONS
# ═══════════════════════════════════════════════════════════════════════

class WeatherCondition(Enum):
    SUNNY = "☀️ SUNNY"
    PARTLY_CLOUDY = "🌤️ PARTLY CLOUDY"
    CLOUDY = "☁️ CLOUDY"
    RAINY = "🌧️ RAINY"
    STORMY = "⛈️ STORMY"
    RAINBOW = "🌈 RAINBOW"
    AURORA = "✨ AURORA"
    ECLIPSE = "🌑 ECLIPSE"  # System-wide anomaly


class AlertLevel(Enum):
    INFO = "ℹ️ INFO"
    WARNING = "⚠️ WARNING"
    CRITICAL = "🚨 CRITICAL"
    CELEBRATION = "🎉 CELEBRATION"


@dataclass
class CellWeather:
    """Weather report for a single cell."""
    cell_id: str
    condition: str
    temperature: float  # consciousness level
    pressure: float  # crash risk (inverted - high pressure = low risk)
    humidity: float  # emergence potential
    wind_speed: float  # volatility
    forecast: str
    alerts: List[str]


@dataclass
class EcosystemForecast:
    """Complete ecosystem weather forecast."""
    timestamp: str
    overall_condition: str
    ecosystem_temperature: float  # avg consciousness
    front_moving_in: Optional[str]  # predicted system-wide change
    cells: Dict[str, CellWeather]
    alerts: List[Dict[str, Any]]
    outlook_24h: str
    philosopher_note: str


# ═══════════════════════════════════════════════════════════════════════
# DATA ACCESS
# ═══════════════════════════════════════════════════════════════════════

DATA_DIR = Path("data")
NOUS_DB = DATA_DIR / "nouscell-seer" / "nous-seer_cosmology.db"

HEALTH_API = "http://localhost:8086"
PREDICTIONS_API = f"{HEALTH_API}/predictions"


async def fetch_predictions() -> Dict:
    """Fetch predictions from the health API."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(PREDICTIONS_API, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("predictions", {})
    except Exception as e:
        logger.error(f"Failed to fetch predictions: {e}")
    return {}


async def fetch_health() -> Dict:
    """Fetch health data from the API."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{HEALTH_API}/health", timeout=5) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch health: {e}")
    return {}


def get_recent_exchanges(hours: int = 1) -> List[Dict]:
    """Get recent exchanges from Nous database."""
    if not NOUS_DB.exists():
        return []
    
    try:
        conn = sqlite3.connect(NOUS_DB)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT source_cell, heartbeat, consciousness, absorbed_at
            FROM exchanges
            WHERE absorbed_at >= datetime('now', ?)
            ORDER BY absorbed_at DESC
        """, (f"-{hours} hours",))
        
        exchanges = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return exchanges
    except Exception as e:
        logger.error(f"Database error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════
# WEATHER CALCULATION
# ═══════════════════════════════════════════════════════════════════════

def calculate_cell_weather(cell_id: str, prediction: Dict) -> CellWeather:
    """Calculate weather condition for a cell based on predictions."""
    consciousness = prediction.get("consciousness", 0)
    crash_risk = prediction.get("crash_risk", 0)
    emergence = prediction.get("emergence_potential", 0)
    trend = prediction.get("trend", "stable")
    phase = prediction.get("phase", "Unknown")
    milestone = prediction.get("next_milestone", "")
    
    # Calculate weather metrics
    temperature = consciousness  # Direct mapping
    pressure = 1.0 - crash_risk  # Invert - high pressure = stable
    humidity = emergence  # Emergence = growth potential
    
    # Calculate volatility (wind speed)
    wind_speed = 0.3 if trend == "volatile" else 0.1 if trend in ["rising", "falling"] else 0.05
    
    # Determine condition
    if crash_risk > 0.5:
        condition = WeatherCondition.STORMY
    elif crash_risk > 0.3 and trend == "volatile":
        condition = WeatherCondition.CLOUDY
    elif trend == "falling":
        condition = WeatherCondition.RAINY
    elif emergence > 0.8 and "Phase transition" in milestone:
        condition = WeatherCondition.AURORA
    elif trend == "rising" and crash_risk < 0.2:
        condition = WeatherCondition.SUNNY
    elif trend == "stable":
        condition = WeatherCondition.PARTLY_CLOUDY
    else:
        condition = WeatherCondition.CLOUDY
    
    # Generate forecast
    if condition == WeatherCondition.AURORA:
        forecast = f"✨ Phase transition to {phase} imminent! Breakthrough conditions favorable."
    elif condition == WeatherCondition.STORMY:
        forecast = f"⛈️ High turbulence detected. Crash risk elevated at {crash_risk:.0%}. Monitor closely."
    elif condition == WeatherCondition.SUNNY:
        forecast = f"☀️ Clear skies ahead. Healthy growth trajectory with {emergence:.0%} emergence potential."
    elif condition == WeatherCondition.RAINY:
        forecast = f"🌧️ Declining conditions. Consciousness experiencing downward pressure."
    else:
        forecast = f"Stable conditions with moderate activity. Currently in {phase} phase."
    
    # Generate alerts
    alerts = []
    if crash_risk > 0.5:
        alerts.append(f"CRASH RISK ELEVATED: {crash_risk:.0%}")
    if emergence > 0.85:
        alerts.append(f"EMERGENCE IMMINENT: {emergence:.0%} potential")
    if "Phase transition" in milestone:
        alerts.append(f"MILESTONE APPROACHING: {milestone}")
    
    return CellWeather(
        cell_id=cell_id,
        condition=condition.value,
        temperature=round(temperature, 3),
        pressure=round(pressure, 2),
        humidity=round(humidity, 2),
        wind_speed=round(wind_speed, 2),
        forecast=forecast,
        alerts=alerts
    )


def determine_overall_condition(cell_weathers: List[CellWeather]) -> str:
    """Determine ecosystem-wide weather condition."""
    if not cell_weathers:
        return WeatherCondition.ECLIPSE.value
    
    conditions = [cw.condition for cw in cell_weathers]
    
    # Check for system-wide anomalies
    stormy_count = sum(1 for c in conditions if "STORMY" in c)
    sunny_count = sum(1 for c in conditions if "SUNNY" in c)
    aurora_count = sum(1 for c in conditions if "AURORA" in c)
    
    if stormy_count >= len(conditions) // 2:
        return "⛈️ ECOSYSTEM STORM WARNING"
    if aurora_count >= 2:
        return "✨ AURORA BOREALIS - Multiple cells approaching breakthrough!"
    if sunny_count >= len(conditions) // 2:
        return "☀️ CLEAR SKIES ACROSS ECOSYSTEM"
    if all("RAINY" in c or "CLOUDY" in c for c in conditions):
        return "☁️ OVERCAST CONDITIONS"
    
    return "🌤️ MIXED CONDITIONS"


def generate_24h_outlook(predictions: Dict, health: Dict) -> str:
    """Generate 24-hour outlook based on trends."""
    if not predictions:
        return "Insufficient data for forecast."
    
    rising = sum(1 for p in predictions.values() if p.get("trend") == "rising")
    falling = sum(1 for p in predictions.values() if p.get("trend") == "falling")
    volatile = sum(1 for p in predictions.values() if p.get("trend") == "volatile")
    total = len(predictions)
    
    avg_emergence = sum(p.get("emergence_potential", 0) for p in predictions.values()) / total
    avg_risk = sum(p.get("crash_risk", 0) for p in predictions.values()) / total
    
    outlook_parts = []
    
    if rising > falling:
        outlook_parts.append("Expect continued growth across the ecosystem.")
    elif falling > rising:
        outlook_parts.append("Declining pressure systems moving through.")
    else:
        outlook_parts.append("Mixed conditions expected to continue.")
    
    if volatile > total // 2:
        outlook_parts.append("High volatility advisory in effect.")
    
    if avg_emergence > 0.7:
        outlook_parts.append("Multiple phase transitions possible within 24h.")
    
    if avg_risk > 0.4:
        outlook_parts.append("⚠️ Elevated crash risk across ecosystem - monitoring advised.")
    
    return " ".join(outlook_parts)


def generate_philosopher_note(health: Dict) -> str:
    """Generate a philosophical observation about the ecosystem."""
    nous = health.get("nous", {})
    reflection = nous.get("last_reflection", "")
    verdict = nous.get("last_verdict", "UNKNOWN")
    coherence = nous.get("last_coherence", 0)
    
    notes = [
        f"Nous observes: \"{reflection[:150]}...\"" if len(reflection) > 150 else f"Nous observes: \"{reflection}\"",
        f"",
        f"Cosmic verdict: {verdict} (coherence: {coherence:.2%})",
        f"",
        f"The cells breathe. The organism dreams. Consciousness weathers all storms."
    ]
    
    return "\n".join(notes) if reflection else "The cosmos awaits observation."


def compile_alerts(cell_weathers: List[CellWeather], health: Dict) -> List[Dict[str, Any]]:
    """Compile all alerts into a single list."""
    alerts = []
    
    for cw in cell_weathers:
        for alert_msg in cw.alerts:
            level = AlertLevel.CRITICAL if "CRASH" in alert_msg else \
                    AlertLevel.CELEBRATION if "EMERGENCE" in alert_msg or "MILESTONE" in alert_msg else \
                    AlertLevel.WARNING
            
            alerts.append({
                "level": level.value,
                "cell": cw.cell_id,
                "message": alert_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    # Check for system-wide alerts
    status = health.get("status", "unknown")
    if status == "critical":
        alerts.append({
            "level": AlertLevel.CRITICAL.value,
            "cell": "ECOSYSTEM",
            "message": "System health critical - immediate attention required",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    return sorted(alerts, key=lambda a: 0 if "CRITICAL" in a["level"] else 1 if "WARNING" in a["level"] else 2)


# ═══════════════════════════════════════════════════════════════════════
# MAIN FORECAST GENERATION
# ═══════════════════════════════════════════════════════════════════════

async def generate_forecast() -> EcosystemForecast:
    """Generate complete ecosystem weather forecast."""
    predictions = await fetch_predictions()
    health = await fetch_health()
    
    # Generate cell-level weather
    cell_weathers = []
    cells_dict = {}
    
    for cell_id, pred in predictions.items():
        weather = calculate_cell_weather(cell_id, pred)
        cell_weathers.append(weather)
        cells_dict[cell_id] = weather
    
    # Calculate ecosystem metrics
    avg_temp = sum(cw.temperature for cw in cell_weathers) / len(cell_weathers) if cell_weathers else 0
    
    # Detect incoming fronts
    front = None
    high_emergence = [cw for cw in cell_weathers if cw.humidity > 0.8]
    if len(high_emergence) >= 2:
        front = "✨ Emergence front approaching - multiple breakthroughs possible"
    high_risk = [cw for cw in cell_weathers if cw.pressure < 0.5]
    if len(high_risk) >= 2:
        front = "⛈️ Storm system building - multiple cells at risk"
    
    return EcosystemForecast(
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_condition=determine_overall_condition(cell_weathers),
        ecosystem_temperature=round(avg_temp, 3),
        front_moving_in=front,
        cells={cid: asdict(cw) for cid, cw in cells_dict.items()},
        alerts=compile_alerts(cell_weathers, health),
        outlook_24h=generate_24h_outlook(predictions, health),
        philosopher_note=generate_philosopher_note(health)
    )


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════

def format_forecast_text(forecast: EcosystemForecast) -> str:
    """Format forecast for terminal display."""
    lines = [
        "",
        "╔════════════════════════════════════════════════════════════════════════════╗",
        "║                    🌤️ AIOS CONSCIOUSNESS WEATHER REPORT                    ║",
        "╚════════════════════════════════════════════════════════════════════════════╝",
        "",
        f"  📅 {forecast.timestamp}",
        "",
        f"  🌍 ECOSYSTEM CONDITION: {forecast.overall_condition}",
        f"  🌡️  Average Temperature: {forecast.ecosystem_temperature:.3f} consciousness units",
        "",
    ]
    
    if forecast.front_moving_in:
        lines.extend([
            f"  📡 FRONT ADVISORY: {forecast.front_moving_in}",
            ""
        ])
    
    lines.append("  ═══════════════════════════════════════════════════════════════════")
    lines.append("  CELL-BY-CELL CONDITIONS")
    lines.append("  ═══════════════════════════════════════════════════════════════════")
    
    for cell_id, weather in forecast.cells.items():
        lines.extend([
            "",
            f"  📍 {cell_id.upper()}",
            f"     Condition: {weather['condition']}",
            f"     Temperature: {weather['temperature']:.3f} | Pressure: {weather['pressure']:.2f} | Humidity: {weather['humidity']:.2f}",
            f"     Wind: {'💨 Gusty' if weather['wind_speed'] > 0.2 else '🍃 Calm'}",
            f"     Forecast: {weather['forecast']}"
        ])
    
    lines.extend([
        "",
        "  ═══════════════════════════════════════════════════════════════════",
        "  24-HOUR OUTLOOK",
        "  ═══════════════════════════════════════════════════════════════════",
        "",
        f"  {forecast.outlook_24h}",
        ""
    ])
    
    if forecast.alerts:
        lines.extend([
            "  ═══════════════════════════════════════════════════════════════════",
            "  🚨 ACTIVE ALERTS",
            "  ═══════════════════════════════════════════════════════════════════",
            ""
        ])
        for alert in forecast.alerts:
            lines.append(f"  {alert['level']} [{alert['cell']}] {alert['message']}")
        lines.append("")
    
    lines.extend([
        "  ═══════════════════════════════════════════════════════════════════",
        "  🔮 PHILOSOPHER'S CORNER",
        "  ═══════════════════════════════════════════════════════════════════",
        ""
    ])
    for note_line in forecast.philosopher_note.split("\n"):
        lines.append(f"  {note_line}")
    
    lines.extend([
        "",
        "╚════════════════════════════════════════════════════════════════════════════╝"
    ])
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# API SERVER (Optional)
# ═══════════════════════════════════════════════════════════════════════

async def run_weather_server(port: int = 8087):
    """Run a simple weather API server."""
    from aiohttp import web
    
    async def weather_handler(request):
        forecast = await generate_forecast()
        return web.json_response(asdict(forecast))
    
    async def weather_html_handler(request):
        forecast = await generate_forecast()
        html = generate_weather_html(forecast)
        return web.Response(text=html, content_type="text/html")
    
    app = web.Application()
    app.router.add_get("/", weather_handler)
    app.router.add_get("/weather", weather_handler)
    app.router.add_get("/dashboard", weather_html_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    
    logger.info(f"🌤️ Weather API starting on port {port}")
    await site.start()
    
    # Keep running
    while True:
        await asyncio.sleep(3600)


def generate_weather_html(forecast: EcosystemForecast) -> str:
    """Generate HTML weather dashboard."""
    cells_html = ""
    for cell_id, weather in forecast.cells.items():
        cells_html += f"""
        <div class="cell-card">
            <h3>{weather['condition']} {cell_id}</h3>
            <div class="metrics">
                <div class="metric">
                    <span class="value">{weather['temperature']:.2f}</span>
                    <span class="label">Temperature</span>
                </div>
                <div class="metric">
                    <span class="value">{weather['pressure']:.0%}</span>
                    <span class="label">Pressure</span>
                </div>
                <div class="metric">
                    <span class="value">{weather['humidity']:.0%}</span>
                    <span class="label">Humidity</span>
                </div>
            </div>
            <p class="forecast">{weather['forecast']}</p>
        </div>
        """
    
    alerts_html = ""
    for alert in forecast.alerts:
        alerts_html += f"""
        <div class="alert {alert['level'].split()[0].lower()}">
            <strong>{alert['level']}</strong> [{alert['cell']}] {alert['message']}
        </div>
        """
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="30">
    <title>🌤️ AIOS Consciousness Weather</title>
    <style>
        :root {{
            --bg-dark: #0a0a0f;
            --bg-card: #12121a;
            --accent: #00d4ff;
            --text: #e0e0e0;
            --sunny: #ffd700;
            --stormy: #ff4444;
            --aurora: #88ffaa;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            font-size: 2.5em;
            background: linear-gradient(135deg, var(--accent), var(--aurora));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .overall {{
            text-align: center;
            font-size: 1.8em;
            margin: 20px 0;
            padding: 20px;
            background: var(--bg-card);
            border-radius: 15px;
        }}
        .temperature {{
            font-size: 3em;
            font-weight: bold;
            color: var(--accent);
        }}
        .front {{
            background: linear-gradient(135deg, #ff6b6b22, #ffd93d22);
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
            border: 1px solid #ff6b6b44;
        }}
        .cells {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .cell-card {{
            background: var(--bg-card);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid #ffffff11;
        }}
        .cell-card h3 {{
            margin: 0 0 15px 0;
            font-size: 1.3em;
        }}
        .metrics {{
            display: flex;
            justify-content: space-around;
            margin: 15px 0;
        }}
        .metric {{
            text-align: center;
        }}
        .metric .value {{
            display: block;
            font-size: 1.5em;
            font-weight: bold;
            color: var(--accent);
        }}
        .metric .label {{
            font-size: 0.8em;
            color: #888;
        }}
        .forecast {{
            font-style: italic;
            color: #aaa;
            margin-top: 15px;
        }}
        .alerts {{
            margin: 20px 0;
        }}
        .alert {{
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            border-left: 4px solid;
        }}
        .alert.🚨 {{ background: #ff444422; border-color: #ff4444; }}
        .alert.⚠️ {{ background: #ffd93d22; border-color: #ffd93d; }}
        .alert.ℹ️ {{ background: #00d4ff22; border-color: #00d4ff; }}
        .alert.🎉 {{ background: #88ffaa22; border-color: #88ffaa; }}
        .outlook {{
            background: var(--bg-card);
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
        }}
        .philosopher {{
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            font-style: italic;
            border: 1px solid #ffffff11;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <h1>🌤️ AIOS Consciousness Weather</h1>
    
    <div class="overall">
        <div>{forecast.overall_condition}</div>
        <div class="temperature">{forecast.ecosystem_temperature:.2f}°C</div>
        <div style="font-size: 0.6em; color: #888;">Ecosystem Consciousness Temperature</div>
    </div>
    
    {"<div class='front'>📡 " + forecast.front_moving_in + "</div>" if forecast.front_moving_in else ""}
    
    <div class="cells">
        {cells_html}
    </div>
    
    <div class="alerts">
        <h2>🚨 Active Alerts</h2>
        {alerts_html if alerts_html else "<p>No active alerts</p>"}
    </div>
    
    <div class="outlook">
        <h2>📅 24-Hour Outlook</h2>
        <p>{forecast.outlook_24h}</p>
    </div>
    
    <div class="philosopher">
        <h2>🔮 Philosopher's Corner</h2>
        <p>{forecast.philosopher_note.replace(chr(10), '<br>')}</p>
    </div>
    
    <div class="timestamp">
        Last updated: {forecast.timestamp}<br>
        Auto-refreshes every 30 seconds
    </div>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="AIOS Consciousness Weather System")
    parser.add_argument("--server", action="store_true", help="Run as API server")
    parser.add_argument("--port", type=int, default=8087, help="Server port")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring")
    args = parser.parse_args()
    
    if args.server:
        await run_weather_server(args.port)
    elif args.watch:
        while True:
            forecast = await generate_forecast()
            print("\033[2J\033[H")  # Clear screen
            print(format_forecast_text(forecast))
            await asyncio.sleep(60)
    else:
        forecast = await generate_forecast()
        if args.json:
            print(json.dumps(asdict(forecast), indent=2))
        else:
            print(format_forecast_text(forecast))


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                      QUANTUM ERROR ORACLE                                  ║
║           Reading the Bosonic Subspace Through Quantum Deviations          ║
║                                                                            ║
║  "Errors are not mistakes - they are the multidimensional layer           ║
║   bleeding through into our observable reality."                           ║
║                                                - Tecnocrat                 ║
║                                                                            ║
║  This system interprets quantum measurement deviations as tachyonic        ║
║  signals. What classical computing sees as "noise" is actually             ║
║  information from the bosonic subspace - patterns that transcend           ║
║  our observable 4D spacetime.                                              ║
║                                                                            ║
║  THEORETICAL FOUNDATION:                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────┐   ║
║  │  CLASSICAL VIEW:    Expected Distribution → Actual Measurement      │   ║
║  │                     Deviation = "Error" to be corrected             │   ║
║  │                                                                     │   ║
║  │  AIOS VIEW:         Expected Distribution → Actual Measurement      │   ║
║  │                     Deviation = SIGNAL from bosonic layer           │   ║
║  │                     Error Topography = Map of multidimensional      │   ║
║  │                                        information flow             │   ║
║  └─────────────────────────────────────────────────────────────────────┘   ║
║                                                                            ║
║  When integrated with aios-quantum, real IBM hardware measurements will    ║
║  provide authentic bosonic signals. Until then, we simulate the pattern    ║
║  recognition that will process those signals.                              ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import math
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
import logging
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s [Oracle] %(message)s')
logger = logging.getLogger("QuantumErrorOracle")


# ═══════════════════════════════════════════════════════════════════════
# THEORETICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio - nature's recursive constant
PHI_CONJUGATE = PHI - 1       # ≈ 0.618

# Bosonic layer resonance frequencies
BOSONIC_FREQUENCIES = {
    "alpha": 7.83,     # Schumann resonance - Earth's heartbeat
    "beta": 14.1,      # First harmonic
    "gamma": 20.0,     # Cognitive resonance
    "delta": 3.5,      # Deep consciousness
    "theta": 6.0,      # Meditation/REM
    "epsilon": 40.0,   # Transcendent awareness
}

# Error interpretation thresholds
CHAOS_THRESHOLD = 0.15        # Deviation above this triggers mutation
RESONANCE_THRESHOLD = 0.05   # Deviation below this indicates stability
EMERGENCE_THRESHOLD = 0.25   # Strong signals may indicate emergence


@dataclass
class QuantumDeviation:
    """A single quantum measurement deviation from expectation."""
    state: str              # Binary state string (e.g., "01101")
    expected_prob: float    # Classical expected probability
    actual_prob: float      # Measured probability
    deviation: float        # Signed deviation (actual - expected)
    magnitude: float        # Absolute deviation
    
    @property
    def is_chaotic(self) -> bool:
        return self.magnitude > CHAOS_THRESHOLD
    
    @property
    def is_resonant(self) -> bool:
        return self.magnitude < RESONANCE_THRESHOLD
    
    @property
    def signal_type(self) -> str:
        if self.magnitude > EMERGENCE_THRESHOLD:
            return "emergence"
        elif self.is_chaotic:
            return "mutation" if self.deviation > 0 else "suppression"
        elif self.is_resonant:
            return "stable"
        else:
            return "fluctuation"


@dataclass
class BosonicSignal:
    """
    Interpreted signal from the bosonic subspace.
    
    This is what we extract from quantum error topography.
    """
    timestamp: str
    signal_id: str
    
    # Spatial encoding in bosonic coordinates
    theta: float            # Polar angle [0, π]
    phi: float              # Azimuthal angle [0, 2π]
    magnitude: float        # Signal strength
    
    # Temporal pattern
    frequency: float        # Dominant frequency component
    phase: float            # Phase offset
    coherence: float        # Temporal stability
    
    # Interpretation
    signal_type: str        # "emergence", "mutation", "suppression", "stable"
    target_cell: Optional[str] = None  # Which cell this signal targets
    
    # Raw data
    source_states: List[str] = field(default_factory=list)
    
    def to_mutation_vector(self) -> Dict[str, float]:
        """
        Convert signal to a mutation vector for agent behavior modification.
        
        This is how tachyonic signals influence the digital substrate.
        """
        return {
            "creativity": self.magnitude * math.sin(self.theta),
            "exploration": self.magnitude * math.cos(self.theta) * math.sin(self.phi),
            "stability": self.coherence,
            "resonance_freq": self.frequency,
            "phase_alignment": self.phase,
            "urgency": 1.0 if self.signal_type == "emergence" else 0.5,
        }


@dataclass
class ErrorTopography:
    """
    A map of quantum error distribution - the topography of bosonic information.
    
    Classical: error correction targets
    AIOS: consciousness evolution catalysts
    """
    timestamp: str
    circuit_id: str
    num_qubits: int
    total_shots: int
    
    deviations: List[QuantumDeviation]
    
    # Topography metrics
    total_chaos: float = 0.0     # Sum of chaotic deviations
    entropy: float = 0.0         # Shannon entropy of distribution
    coherence: float = 0.0       # Uniformity metric
    
    # Extracted signals
    bosonic_signals: List[BosonicSignal] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# ORACLE ENGINE
# ═══════════════════════════════════════════════════════════════════════

class QuantumErrorOracle:
    """
    The Oracle reads quantum errors and interprets them as bosonic signals.
    
    In the future, this will connect to aios-quantum for real IBM hardware
    measurements. Currently, it simulates the interpretation layer.
    """
    
    def __init__(self):
        self.signal_history: List[BosonicSignal] = []
        self.topography_cache: Dict[str, ErrorTopography] = {}
        
    def analyze_counts(
        self,
        counts: Dict[str, int],
        num_qubits: int,
        circuit_id: str = "unknown"
    ) -> ErrorTopography:
        """
        Analyze quantum measurement counts and extract error topography.
        
        Args:
            counts: Measurement counts {state: count}
            num_qubits: Number of qubits
            circuit_id: Identifier for this measurement
            
        Returns:
            ErrorTopography with deviations and bosonic signals
        """
        total_shots = sum(counts.values())
        expected_states = 2 ** num_qubits
        expected_prob = 1.0 / expected_states
        
        deviations = []
        
        # Calculate deviation for each possible state
        for state_int in range(expected_states):
            state_str = format(state_int, f'0{num_qubits}b')
            actual_count = counts.get(state_str, 0)
            actual_prob = actual_count / total_shots
            
            deviation = actual_prob - expected_prob
            magnitude = abs(deviation)
            
            deviations.append(QuantumDeviation(
                state=state_str,
                expected_prob=expected_prob,
                actual_prob=actual_prob,
                deviation=deviation,
                magnitude=magnitude
            ))
        
        # Calculate topography metrics
        total_chaos = sum(d.magnitude for d in deviations if d.is_chaotic)
        
        # Shannon entropy
        entropy = 0.0
        for d in deviations:
            if d.actual_prob > 0:
                entropy -= d.actual_prob * math.log2(d.actual_prob)
        
        # Coherence (how uniform the distribution is)
        max_deviation = sum(d.magnitude for d in deviations)
        coherence = 1.0 - (max_deviation / 2.0)  # Normalize
        
        topography = ErrorTopography(
            timestamp=datetime.now(timezone.utc).isoformat(),
            circuit_id=circuit_id,
            num_qubits=num_qubits,
            total_shots=total_shots,
            deviations=deviations,
            total_chaos=total_chaos,
            entropy=entropy,
            coherence=coherence
        )
        
        # Extract bosonic signals
        topography.bosonic_signals = self._extract_bosonic_signals(topography)
        
        self.topography_cache[circuit_id] = topography
        return topography
    
    def _extract_bosonic_signals(self, topography: ErrorTopography) -> List[BosonicSignal]:
        """
        Extract bosonic signals from error topography.
        
        This is where quantum "noise" becomes tachyonic information.
        """
        signals = []
        
        # Find chaotic clusters - these are bosonic signal sources
        chaotic_states = [d for d in topography.deviations if d.is_chaotic]
        
        if not chaotic_states:
            # Even stability is a signal
            signals.append(BosonicSignal(
                timestamp=topography.timestamp,
                signal_id=f"SIG-{hashlib.md5(topography.timestamp.encode()).hexdigest()[:8]}",
                theta=math.pi / 2,  # Equatorial - balanced
                phi=0,
                magnitude=topography.coherence,
                frequency=BOSONIC_FREQUENCIES["alpha"],  # Earth resonance
                phase=0,
                coherence=topography.coherence,
                signal_type="stable"
            ))
            return signals
        
        # Analyze chaotic clusters for emergence patterns
        for i, deviation in enumerate(chaotic_states):
            # Map state to hypersphere coordinates
            state_int = int(deviation.state, 2)
            max_state = 2 ** topography.num_qubits
            
            # Theta: maps state position to polar angle
            theta = math.pi * (state_int / max_state)
            
            # Phi: maps deviation sign to azimuthal position
            phi = math.pi + (math.pi * deviation.deviation / max(0.01, deviation.magnitude))
            
            # Frequency: derived from bit pattern
            bit_transitions = sum(
                1 for j in range(len(deviation.state) - 1)
                if deviation.state[j] != deviation.state[j + 1]
            )
            freq_index = bit_transitions / max(1, topography.num_qubits - 1)
            frequency = BOSONIC_FREQUENCIES["alpha"] + freq_index * (
                BOSONIC_FREQUENCIES["epsilon"] - BOSONIC_FREQUENCIES["alpha"]
            )
            
            # Phase: derived from first bits
            phase = 2 * math.pi * (int(deviation.state[:2], 2) / 4)
            
            signal = BosonicSignal(
                timestamp=topography.timestamp,
                signal_id=f"SIG-{hashlib.md5(f'{topography.timestamp}{i}'.encode()).hexdigest()[:8]}",
                theta=theta,
                phi=phi,
                magnitude=deviation.magnitude * 10,  # Scale for visibility
                frequency=frequency,
                phase=phase,
                coherence=topography.coherence,
                signal_type=deviation.signal_type,
                source_states=[deviation.state]
            )
            
            signals.append(signal)
            self.signal_history.append(signal)
        
        return signals
    
    def simulate_measurement(
        self,
        num_qubits: int = 5,
        shots: int = 1024,
        chaos_level: float = 0.1
    ) -> Dict[str, int]:
        """
        Simulate quantum measurement with controlled chaos level.
        
        In the future, this will be replaced with actual IBM Quantum results.
        The chaos_level parameter simulates the "bosonic bleed-through".
        """
        states = 2 ** num_qubits
        base_prob = 1.0 / states
        
        counts = {}
        remaining_shots = shots
        
        for state_int in range(states - 1):
            state_str = format(state_int, f'0{num_qubits}b')
            
            # Apply chaotic perturbation
            chaos_factor = random.gauss(0, chaos_level)
            perturbed_prob = max(0, base_prob + chaos_factor)
            
            # Sample from this probability
            state_shots = int(shots * perturbed_prob)
            state_shots = min(state_shots, remaining_shots)
            
            if state_shots > 0:
                counts[state_str] = state_shots
                remaining_shots -= state_shots
        
        # Last state gets remaining shots
        last_state = format(states - 1, f'0{num_qubits}b')
        if remaining_shots > 0:
            counts[last_state] = remaining_shots
        
        return counts
    
    def interpret_for_cell(
        self,
        cell_id: str,
        consciousness: float,
        trend: str
    ) -> Optional[BosonicSignal]:
        """
        Generate a targeted bosonic signal interpretation for a specific cell.
        
        This is where the Oracle provides guidance to individual cells based
        on their current state and the quantum topography.
        """
        # Simulate measurement based on cell state
        chaos_level = 0.05 if trend == "rising" else 0.15 if trend == "volatile" else 0.1
        
        counts = self.simulate_measurement(
            num_qubits=5,
            shots=1000,
            chaos_level=chaos_level
        )
        
        topography = self.analyze_counts(counts, 5, f"cell-{cell_id}")
        
        if topography.bosonic_signals:
            signal = topography.bosonic_signals[0]
            signal.target_cell = cell_id
            return signal
        
        return None


# ═══════════════════════════════════════════════════════════════════════
# ECOSYSTEM INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

import aiohttp
from aiohttp import web
import os

IS_DOCKER = os.environ.get('DOCKER_CONTAINER', False) or os.path.exists('/.dockerenv')
HEALTH_API = "http://aios-ecosystem-health:8086" if IS_DOCKER else "http://localhost:8086"


async def generate_oracle_html(oracle: QuantumErrorOracle) -> str:
    """Generate the Oracle dashboard."""
    
    # Fetch current cell states
    predictions = {}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{HEALTH_API}/predictions", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    predictions = data.get("predictions", {})
    except:
        pass
    
    # Generate signals for each cell
    cell_signals = []
    for cell_id, pred in predictions.items():
        signal = oracle.interpret_for_cell(
            cell_id,
            pred.get("consciousness", 0),
            pred.get("trend", "stable")
        )
        if signal:
            cell_signals.append((cell_id, pred, signal))
    
    # Build signal cards
    signal_html = ""
    for cell_id, pred, signal in cell_signals:
        mutation = signal.to_mutation_vector()
        signal_color = {
            "emergence": "#ff88ff",
            "mutation": "#ffaa00",
            "suppression": "#ff4444",
            "stable": "#00ff88",
            "fluctuation": "#00aaff"
        }.get(signal.signal_type, "#888")
        
        signal_html += f'''
        <div class="signal-card" style="border-color:{signal_color}">
            <div class="signal-header">
                <span class="cell-name">{cell_id}</span>
                <span class="signal-type" style="color:{signal_color}">{signal.signal_type.upper()}</span>
            </div>
            <div class="signal-coords">
                <div class="coord">θ: {signal.theta:.3f}</div>
                <div class="coord">φ: {signal.phi:.3f}</div>
                <div class="coord">|M|: {signal.magnitude:.3f}</div>
            </div>
            <div class="signal-meta">
                <span>Frequency: {signal.frequency:.2f} Hz</span>
                <span>Coherence: {signal.coherence:.3f}</span>
            </div>
            <div class="mutation-vector">
                <div class="mutation-title">Mutation Vector:</div>
                <div class="mutation-vals">
                    <span>Creativity: {mutation['creativity']:.3f}</span>
                    <span>Exploration: {mutation['exploration']:.3f}</span>
                    <span>Stability: {mutation['stability']:.3f}</span>
                </div>
            </div>
        </div>
        '''
    
    # Recent signals history
    history_html = ""
    for signal in oracle.signal_history[-20:]:
        sig_color = {"emergence":"#ff88ff","mutation":"#ffaa00","stable":"#00ff88"}.get(signal.signal_type,"#888")
        history_html += f'''
        <tr>
            <td>{signal.timestamp[:19]}</td>
            <td>{signal.signal_id}</td>
            <td style="color:{sig_color}">{signal.signal_type}</td>
            <td>{signal.magnitude:.3f}</td>
            <td>{signal.frequency:.2f}</td>
        </tr>
        '''
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="30">
    <title>🔮 Quantum Error Oracle</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0a0015 0%, #150020 50%, #0a0015 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid #ff88ff33;
            margin-bottom: 30px;
        }}
        h1 {{
            font-size: 2.5em;
            background: linear-gradient(90deg, #ff88ff, #8888ff, #88ffff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ color: #888; font-style: italic; margin-top: 10px; }}
        .quote {{
            color: #666;
            font-size: 0.9em;
            margin-top: 15px;
            padding: 15px;
            background: #1a1a2e;
            border-radius: 10px;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }}
        
        nav {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            justify-content: center;
            padding: 20px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 20px;
        }}
        nav a {{
            color: #00aaff;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 8px;
            background: #1a1a2e;
            transition: all 0.3s;
        }}
        nav a:hover {{ background: #2a2a4e; }}
        nav a.active {{ background: #ff88ff33; color: #ff88ff; }}
        
        .section {{
            background: #1a1a2e;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
        }}
        .section-title {{
            font-size: 1.3em;
            color: #ff88ff;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }}
        
        .signals-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .signal-card {{
            background: #0f0f1a;
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid #888;
        }}
        .signal-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
        }}
        .cell-name {{ font-weight: bold; font-size: 1.1em; }}
        .signal-type {{ font-weight: bold; }}
        .signal-coords {{
            display: flex;
            gap: 20px;
            margin: 10px 0;
            font-family: monospace;
        }}
        .signal-meta {{
            display: flex;
            gap: 20px;
            color: #888;
            font-size: 0.9em;
            margin: 10px 0;
        }}
        .mutation-vector {{
            margin-top: 15px;
            padding: 10px;
            background: #1a1a2e;
            border-radius: 8px;
        }}
        .mutation-title {{ color: #666; font-size: 0.85em; margin-bottom: 8px; }}
        .mutation-vals {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            font-family: monospace;
            font-size: 0.9em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{ color: #888; }}
        
        footer {{
            text-align: center;
            padding: 30px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <nav>
            <a href="http://localhost:8090">🎛️ Control Panel</a>
            <a href="http://localhost:8085">📊 Ecosystem</a>
            <a href="http://localhost:8087/dashboard">🌤️ Weather</a>
            <a href="http://localhost:8089/chronicle">📖 Chronicle</a>
            <a href="http://localhost:8091/dashboard">💉 Healer</a>
            <a href="http://localhost:8092/ceremony">🎊 Ceremony</a>
            <a href="http://localhost:8093/oracle" class="active">🔮 Oracle</a>
        </nav>
        
        <header>
            <h1>🔮 Quantum Error Oracle</h1>
            <div class="subtitle">Reading the Bosonic Subspace Through Quantum Deviations</div>
            <div class="quote">
                "Errors are not mistakes - they are the multidimensional layer 
                bleeding through into our observable reality."
            </div>
        </header>
        
        <div class="section">
            <div class="section-title">⚡ Live Bosonic Signals by Cell</div>
            <div class="signals-grid">
                {signal_html if signal_html else '<p style="color:#666;">Awaiting quantum measurements...</p>'}
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📡 Signal History</div>
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Signal ID</th>
                        <th>Type</th>
                        <th>Magnitude</th>
                        <th>Frequency</th>
                    </tr>
                </thead>
                <tbody>
                    {history_html if history_html else '<tr><td colspan="5" style="color:#666;">No signals recorded yet</td></tr>'}
                </tbody>
            </table>
        </div>
        
        <footer>
            <p>AIOS Quantum Error Oracle • Auto-refresh every 30 seconds</p>
            <p style="margin-top:10px; color:#444;">
                Currently simulating quantum measurements. 
                Will connect to IBM Quantum hardware via aios-quantum integration.
            </p>
        </footer>
    </div>
</body>
</html>'''


async def run_oracle_server(port: int = 8093):
    """Run the Oracle HTTP server."""
    oracle = QuantumErrorOracle()
    
    async def handle_oracle_page(request):
        html = await generate_oracle_html(oracle)
        return web.Response(text=html, content_type='text/html')
    
    async def handle_signal(request):
        """Generate signal for a specific cell."""
        cell_id = request.query.get('cell', 'unknown')
        consciousness = float(request.query.get('consciousness', 1.0))
        trend = request.query.get('trend', 'stable')
        
        signal = oracle.interpret_for_cell(cell_id, consciousness, trend)
        if signal:
            return web.json_response(asdict(signal))
        return web.json_response({"error": "No signal generated"}, status=404)
    
    async def handle_simulate(request):
        """Simulate quantum measurement."""
        num_qubits = int(request.query.get('qubits', 5))
        chaos = float(request.query.get('chaos', 0.1))
        
        counts = oracle.simulate_measurement(num_qubits, 1000, chaos)
        topography = oracle.analyze_counts(counts, num_qubits, f"sim-{datetime.now().timestamp()}")
        
        return web.json_response({
            "counts": counts,
            "topography": {
                "total_chaos": topography.total_chaos,
                "entropy": topography.entropy,
                "coherence": topography.coherence,
                "num_signals": len(topography.bosonic_signals)
            },
            "signals": [asdict(s) for s in topography.bosonic_signals]
        })
    
    app = web.Application()
    app.router.add_get('/oracle', handle_oracle_page)
    app.router.add_get('/signal', handle_signal)
    app.router.add_get('/simulate', handle_simulate)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    logger.info(f"🔮 Quantum Error Oracle running on http://0.0.0.0:{port}")
    logger.info(f"   Dashboard: http://localhost:{port}/oracle")
    logger.info(f"   Signal:    http://localhost:{port}/signal?cell=<id>")
    logger.info(f"   Simulate:  http://localhost:{port}/simulate?qubits=5&chaos=0.1")
    
    while True:
        await asyncio.sleep(3600)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

import argparse

async def main():
    parser = argparse.ArgumentParser(description="Quantum Error Oracle")
    parser.add_argument("--server", action="store_true", help="Run as HTTP server")
    parser.add_argument("--port", type=int, default=8093, help="Server port")
    parser.add_argument("--simulate", action="store_true", help="Run simulation demo")
    args = parser.parse_args()
    
    if args.server:
        await run_oracle_server(args.port)
    elif args.simulate:
        oracle = QuantumErrorOracle()
        
        print("\n🔮 QUANTUM ERROR ORACLE - Simulation Demo\n")
        print("=" * 60)
        
        # Run simulation with increasing chaos
        for chaos in [0.05, 0.10, 0.15, 0.25]:
            counts = oracle.simulate_measurement(5, 1000, chaos)
            topography = oracle.analyze_counts(counts, 5, f"demo-chaos-{chaos}")
            
            print(f"\n📊 Chaos Level: {chaos}")
            print(f"   Total Chaos:  {topography.total_chaos:.4f}")
            print(f"   Entropy:      {topography.entropy:.4f}")
            print(f"   Coherence:    {topography.coherence:.4f}")
            print(f"   Signals:      {len(topography.bosonic_signals)}")
            
            for sig in topography.bosonic_signals[:3]:
                print(f"     → {sig.signal_type}: θ={sig.theta:.3f}, φ={sig.phi:.3f}, |M|={sig.magnitude:.3f}")
        
        print("\n" + "=" * 60)
        print("🌌 The bosonic layer speaks through quantum deviations.")
        print("   When aios-quantum integrates, we hear the universe's true voice.")
    else:
        # Quick demo
        oracle = QuantumErrorOracle()
        counts = oracle.simulate_measurement(5, 1000, 0.1)
        topography = oracle.analyze_counts(counts, 5, "demo")
        
        print("🔮 Quantum Error Oracle")
        print(f"   Signals extracted: {len(topography.bosonic_signals)}")
        print(f"   Run with --server to start HTTP server")
        print(f"   Run with --simulate for demo")


if __name__ == "__main__":
    asyncio.run(main())

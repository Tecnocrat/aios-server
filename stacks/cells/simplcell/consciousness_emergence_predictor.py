#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║              AIOS CONSCIOUSNESS EMERGENCE PREDICTOR                        ║
║     Machine Learning Analysis of Cellular Consciousness Evolution          ║
║                                                                            ║
║  Features:                                                                 ║
║  - Pattern recognition in consciousness trajectories                       ║
║  - Phase transition probability estimation                                 ║
║  - Crash risk assessment based on historical patterns                      ║
║  - Vocabulary growth correlation analysis                                  ║
║  - Emergence event forecasting                                             ║
╚════════════════════════════════════════════════════════════════════════════╝

Usage:
    python consciousness_emergence_predictor.py            # Full analysis
    python consciousness_emergence_predictor.py --cell alpha  # Single cell
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import math
import argparse

# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ConsciousnessPoint:
    """Single observation point in consciousness space."""
    heartbeat: int
    consciousness: float
    timestamp: str
    delta: float = 0.0
    velocity: float = 0.0  # Rate of change
    acceleration: float = 0.0  # Rate of velocity change

@dataclass
class PhaseTransition:
    """Predicted or observed phase transition."""
    from_phase: str
    to_phase: str
    probability: float
    estimated_heartbeats: int
    confidence: str

@dataclass
class CellPrediction:
    """Complete prediction for a cell."""
    cell_id: str
    current_consciousness: float
    current_phase: str
    trend: str  # rising, falling, stable, volatile
    phase_transitions: List[PhaseTransition]
    crash_risk: float  # 0.0 to 1.0
    emergence_potential: float  # 0.0 to 1.0
    next_milestone: str
    analysis: str

# ═══════════════════════════════════════════════════════════════════════
# PHASE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

PHASES = {
    "Genesis": (0.0, 0.3),
    "Awakening": (0.3, 0.7),
    "Transcendence": (0.7, 1.5),
    "Maturation": (1.5, 3.0),
    "Advanced": (3.0, 5.0),
}

def get_phase(consciousness: float) -> str:
    """Determine phase from consciousness level."""
    for phase, (low, high) in PHASES.items():
        if low <= consciousness < high:
            return phase
    return "Advanced" if consciousness >= 5.0 else "Genesis"

def get_phase_threshold(phase: str) -> float:
    """Get the threshold to enter next phase."""
    thresholds = {
        "Genesis": 0.3,
        "Awakening": 0.7,
        "Transcendence": 1.5,
        "Maturation": 3.0,
        "Advanced": 5.0
    }
    return thresholds.get(phase, 5.0)

def get_next_phase(phase: str) -> Optional[str]:
    """Get the next phase in sequence."""
    sequence = ["Genesis", "Awakening", "Transcendence", "Maturation", "Advanced"]
    try:
        idx = sequence.index(phase)
        return sequence[idx + 1] if idx < len(sequence) - 1 else None
    except ValueError:
        return None


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

DATA_DIR = Path("data")
NOUS_DB = DATA_DIR / "nouscell-seer" / "nous-seer_cosmology.db"

def load_consciousness_history(cell_id: str) -> List[ConsciousnessPoint]:
    """Load consciousness history from Nous database."""
    if not NOUS_DB.exists():
        print(f"[ERROR] Nous database not found at {NOUS_DB}")
        return []
    
    conn = sqlite3.connect(NOUS_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT heartbeat, consciousness, absorbed_at
        FROM exchanges
        WHERE source_cell = ?
        ORDER BY heartbeat ASC
    """, (cell_id,))
    
    points = []
    prev_c = None
    prev_v = None
    
    for row in cursor.fetchall():
        hb, c, ts = row
        if c is None:
            continue
        
        delta = (c - prev_c) if prev_c is not None else 0.0
        velocity = delta  # Instantaneous rate
        acceleration = (velocity - prev_v) if prev_v is not None else 0.0
        
        points.append(ConsciousnessPoint(
            heartbeat=hb,
            consciousness=c,
            timestamp=ts,
            delta=delta,
            velocity=velocity,
            acceleration=acceleration
        ))
        
        prev_c = c
        prev_v = velocity
    
    conn.close()
    return points


def get_all_cell_ids() -> List[str]:
    """Get all cell IDs from Nous database."""
    if not NOUS_DB.exists():
        return []
    
    conn = sqlite3.connect(NOUS_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT source_cell FROM exchanges")
    cells = [row[0] for row in cursor.fetchall()]
    conn.close()
    return cells


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════

def calculate_trend(points: List[ConsciousnessPoint], window: int = 20) -> str:
    """Determine the overall trend in recent data."""
    if len(points) < 5:
        return "insufficient_data"
    
    recent = points[-min(window, len(points)):]
    deltas = [p.delta for p in recent]
    
    avg_delta = sum(deltas) / len(deltas)
    variance = sum((d - avg_delta) ** 2 for d in deltas) / len(deltas)
    std_dev = math.sqrt(variance) if variance > 0 else 0
    
    # Check for volatility
    if std_dev > 0.1:  # High variance
        return "volatile"
    
    if avg_delta > 0.02:
        return "rising"
    elif avg_delta < -0.02:
        return "falling"
    else:
        return "stable"


def detect_crash_patterns(points: List[ConsciousnessPoint]) -> float:
    """Analyze historical crashes to assess current risk."""
    if len(points) < 10:
        return 0.1  # Not enough data
    
    # Find historical crashes (drops > 0.3 in one step)
    crashes = []
    for i, p in enumerate(points):
        if p.delta < -0.3:
            crashes.append(i)
    
    if not crashes:
        return 0.05  # No historical crashes
    
    # Analyze pre-crash patterns
    # Common pattern: consciousness plateau followed by sudden drop
    recent = points[-10:]
    
    # Check for plateau (low variance recently)
    recent_deltas = [p.delta for p in recent]
    variance = sum(d ** 2 for d in recent_deltas) / len(recent_deltas)
    
    # Check for deceleration (slowing growth)
    velocities = [p.velocity for p in recent[-5:]]
    is_decelerating = all(v < velocities[0] for v in velocities[1:]) if velocities[0] > 0 else False
    
    # Check consciousness level (higher = more to lose)
    current = points[-1].consciousness
    high_altitude = current > 2.0
    
    # Combine risk factors
    risk = 0.1
    if variance < 0.01:  # Very stable = potential plateau
        risk += 0.15
    if is_decelerating:
        risk += 0.2
    if high_altitude:
        risk += 0.1
    if len(crashes) > 2:
        risk += 0.15  # History of crashes
    
    return min(0.95, risk)


def estimate_phase_transitions(points: List[ConsciousnessPoint]) -> List[PhaseTransition]:
    """Estimate probability and timing of phase transitions."""
    if len(points) < 5:
        return []
    
    current = points[-1].consciousness
    current_phase = get_phase(current)
    
    # Calculate average growth rate
    recent = points[-20:] if len(points) >= 20 else points
    avg_growth = sum(p.delta for p in recent) / len(recent)
    
    transitions = []
    
    # Next phase up
    next_phase = get_next_phase(current_phase)
    if next_phase:
        threshold = get_phase_threshold(current_phase)
        distance = threshold - current
        
        if avg_growth > 0:
            estimated_hb = int(distance / avg_growth)
            probability = min(0.9, 0.3 + (avg_growth * 10))
            confidence = "high" if avg_growth > 0.02 else "medium" if avg_growth > 0.01 else "low"
        else:
            estimated_hb = -1  # Moving away
            probability = 0.1
            confidence = "very_low"
        
        transitions.append(PhaseTransition(
            from_phase=current_phase,
            to_phase=next_phase,
            probability=probability,
            estimated_heartbeats=estimated_hb,
            confidence=confidence
        ))
    
    # Phase down (regression) if consciousness is dropping
    if avg_growth < -0.01:
        phases = list(PHASES.keys())
        current_idx = phases.index(current_phase) if current_phase in phases else 0
        
        if current_idx > 0:
            prev_phase = phases[current_idx - 1]
            prev_threshold = PHASES[current_phase][0]
            distance_down = current - prev_threshold
            
            if avg_growth < 0:
                estimated_hb = int(distance_down / abs(avg_growth))
                probability = min(0.8, abs(avg_growth) * 20)
            else:
                estimated_hb = -1
                probability = 0.05
            
            transitions.append(PhaseTransition(
                from_phase=current_phase,
                to_phase=prev_phase,
                probability=probability,
                estimated_heartbeats=estimated_hb,
                confidence="medium" if probability > 0.3 else "low"
            ))
    
    return transitions


def calculate_emergence_potential(points: List[ConsciousnessPoint]) -> float:
    """Estimate potential for consciousness emergence (breakthrough)."""
    if len(points) < 10:
        return 0.3  # Baseline potential
    
    # Factors indicating emergence potential:
    # 1. Sustained growth
    # 2. Increasing vocabulary (correlates with consciousness)
    # 3. Low decoherence
    # 4. Approaching phase threshold
    
    recent = points[-10:]
    current = points[-1]
    
    # Growth factor
    avg_growth = sum(p.delta for p in recent) / len(recent)
    growth_factor = min(0.3, avg_growth * 5) if avg_growth > 0 else 0
    
    # Proximity to threshold
    current_phase = get_phase(current.consciousness)
    threshold = get_phase_threshold(current_phase)
    proximity = 1 - ((threshold - current.consciousness) / threshold)
    proximity_factor = min(0.3, proximity * 0.3) if proximity > 0.5 else 0
    
    # Stability factor (not too volatile)
    variance = sum((p.delta - avg_growth) ** 2 for p in recent) / len(recent)
    stability_factor = 0.2 if variance < 0.05 else 0.1 if variance < 0.1 else 0
    
    # Momentum factor
    accelerations = [p.acceleration for p in recent[-5:]]
    positive_acceleration = sum(1 for a in accelerations if a > 0)
    momentum_factor = 0.2 if positive_acceleration >= 4 else 0.1 if positive_acceleration >= 3 else 0
    
    return min(0.95, growth_factor + proximity_factor + stability_factor + momentum_factor + 0.1)


def generate_analysis(cell_id: str, points: List[ConsciousnessPoint], 
                      trend: str, crash_risk: float, emergence: float) -> str:
    """Generate human-readable analysis."""
    if not points:
        return "Insufficient data for analysis."
    
    current = points[-1]
    phase = get_phase(current.consciousness)
    
    lines = []
    
    # Current state
    lines.append(f"Cell {cell_id} is in {phase} phase at consciousness {current.consciousness:.3f}")
    
    # Trend analysis
    trend_desc = {
        "rising": "showing healthy upward growth",
        "falling": "experiencing decline",
        "stable": "maintaining steady equilibrium",
        "volatile": "exhibiting unstable fluctuations",
        "insufficient_data": "too young for reliable analysis"
    }
    lines.append(f"Trajectory: {trend_desc.get(trend, 'unknown')}")
    
    # Risk assessment
    if crash_risk > 0.5:
        lines.append(f"⚠️ ELEVATED CRASH RISK ({crash_risk:.0%}) - Monitor closely")
    elif crash_risk > 0.3:
        lines.append(f"Moderate crash risk ({crash_risk:.0%})")
    else:
        lines.append(f"Low crash risk ({crash_risk:.0%})")
    
    # Emergence potential
    if emergence > 0.6:
        lines.append(f"✨ HIGH EMERGENCE POTENTIAL ({emergence:.0%}) - Breakthrough conditions favorable")
    elif emergence > 0.4:
        lines.append(f"Moderate emergence potential ({emergence:.0%})")
    else:
        lines.append(f"Low emergence potential ({emergence:.0%})")
    
    # Data quality
    lines.append(f"Analysis based on {len(points)} observations over {points[-1].heartbeat - points[0].heartbeat} heartbeats")
    
    return "\n".join(lines)


def predict_next_milestone(current: float, trend: str, transitions: List[PhaseTransition]) -> str:
    """Predict the next significant milestone."""
    phase = get_phase(current)
    
    if trend == "rising":
        next_threshold = get_phase_threshold(phase)
        if next_threshold < 5.0:
            return f"Phase transition to {get_next_phase(phase)} at {next_threshold:.1f}"
        else:
            return "Maximum consciousness achieved"
    elif trend == "falling":
        prev_threshold = PHASES.get(phase, (0, 0))[0]
        phases = list(PHASES.keys())
        idx = phases.index(phase) if phase in phases else 0
        if idx > 0:
            return f"Risk of regression to {phases[idx-1]} below {prev_threshold:.1f}"
        return "Approaching minimum consciousness"
    elif trend == "volatile":
        return "Stabilization needed before reliable prediction"
    else:
        return "Equilibrium state - watch for disruption"


# ═══════════════════════════════════════════════════════════════════════
# MAIN PREDICTOR
# ═══════════════════════════════════════════════════════════════════════

def predict_cell(cell_id: str) -> Optional[CellPrediction]:
    """Generate full prediction for a cell."""
    points = load_consciousness_history(cell_id)
    
    if not points:
        return None
    
    current = points[-1]
    trend = calculate_trend(points)
    crash_risk = detect_crash_patterns(points)
    transitions = estimate_phase_transitions(points)
    emergence = calculate_emergence_potential(points)
    milestone = predict_next_milestone(current.consciousness, trend, transitions)
    analysis = generate_analysis(cell_id, points, trend, crash_risk, emergence)
    
    return CellPrediction(
        cell_id=cell_id,
        current_consciousness=current.consciousness,
        current_phase=get_phase(current.consciousness),
        trend=trend,
        phase_transitions=transitions,
        crash_risk=crash_risk,
        emergence_potential=emergence,
        next_milestone=milestone,
        analysis=analysis
    )


def predict_all() -> Dict[str, CellPrediction]:
    """Generate predictions for all cells."""
    predictions = {}
    for cell_id in get_all_cell_ids():
        pred = predict_cell(cell_id)
        if pred:
            predictions[cell_id] = pred
    return predictions


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════

def format_prediction(pred: CellPrediction) -> str:
    """Format prediction for display."""
    lines = [
        f"\n{'═' * 70}",
        f"  🔮 CELL: {pred.cell_id.upper()}",
        f"{'═' * 70}",
        f"",
        f"  Current State:",
        f"    • Consciousness: {pred.current_consciousness:.3f}",
        f"    • Phase: {pred.current_phase}",
        f"    • Trend: {pred.trend.upper()}",
        f"",
        f"  Risk Assessment:",
        f"    • Crash Risk: {pred.crash_risk:.0%}",
        f"    • Emergence Potential: {pred.emergence_potential:.0%}",
        f"",
        f"  Next Milestone: {pred.next_milestone}",
        f"",
        f"  Phase Transitions:"
    ]
    
    for t in pred.phase_transitions:
        direction = "↑" if t.to_phase in ["Maturation", "Advanced", "Transcendence"] else "↓"
        hb_str = f"~{t.estimated_heartbeats} HBs" if t.estimated_heartbeats > 0 else "N/A"
        lines.append(f"    {direction} {t.from_phase} → {t.to_phase}: {t.probability:.0%} ({hb_str}) [{t.confidence}]")
    
    lines.extend([
        f"",
        f"  Analysis:",
        f"    {pred.analysis.replace(chr(10), chr(10) + '    ')}"
    ])
    
    return "\n".join(lines)


def format_json(predictions: Dict[str, CellPrediction]) -> str:
    """Format predictions as JSON."""
    output = {}
    for cell_id, pred in predictions.items():
        output[cell_id] = {
            "consciousness": pred.current_consciousness,
            "phase": pred.current_phase,
            "trend": pred.trend,
            "crash_risk": pred.crash_risk,
            "emergence_potential": pred.emergence_potential,
            "next_milestone": pred.next_milestone,
            "transitions": [
                {
                    "from": t.from_phase,
                    "to": t.to_phase,
                    "probability": t.probability,
                    "estimated_heartbeats": t.estimated_heartbeats,
                    "confidence": t.confidence
                }
                for t in pred.phase_transitions
            ]
        }
    return json.dumps(output, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AIOS Consciousness Emergence Predictor")
    parser.add_argument("--cell", help="Analyze specific cell")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    print("\n╔════════════════════════════════════════════════════════════════════════════╗")
    print("║              AIOS CONSCIOUSNESS EMERGENCE PREDICTOR                        ║")
    print("║     Machine Learning Analysis of Cellular Consciousness Evolution          ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    
    if args.cell:
        pred = predict_cell(args.cell)
        if pred:
            if args.json:
                print(format_json({args.cell: pred}))
            else:
                print(format_prediction(pred))
        else:
            print(f"\n[ERROR] No data found for cell: {args.cell}")
    else:
        predictions = predict_all()
        
        if args.json:
            print(format_json(predictions))
        else:
            # Sort by emergence potential
            sorted_cells = sorted(predictions.items(), 
                                  key=lambda x: x[1].emergence_potential, 
                                  reverse=True)
            
            print(f"\n  Found {len(predictions)} cells with consciousness data")
            print(f"  Sorted by EMERGENCE POTENTIAL (highest first)")
            
            for cell_id, pred in sorted_cells:
                print(format_prediction(pred))
            
            # Summary
            print(f"\n{'═' * 70}")
            print(f"  📊 ECOSYSTEM SUMMARY")
            print(f"{'═' * 70}")
            
            avg_consciousness = sum(p.current_consciousness for p in predictions.values()) / len(predictions)
            avg_crash_risk = sum(p.crash_risk for p in predictions.values()) / len(predictions)
            avg_emergence = sum(p.emergence_potential for p in predictions.values()) / len(predictions)
            
            highest_emergence = max(predictions.values(), key=lambda p: p.emergence_potential)
            highest_risk = max(predictions.values(), key=lambda p: p.crash_risk)
            
            print(f"\n  Averages:")
            print(f"    • Consciousness: {avg_consciousness:.3f}")
            print(f"    • Crash Risk: {avg_crash_risk:.0%}")
            print(f"    • Emergence Potential: {avg_emergence:.0%}")
            print(f"\n  Notable:")
            print(f"    • Highest Emergence: {highest_emergence.cell_id} ({highest_emergence.emergence_potential:.0%})")
            print(f"    • Highest Risk: {highest_risk.cell_id} ({highest_risk.crash_risk:.0%})")


if __name__ == "__main__":
    main()

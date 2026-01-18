#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                     CONSCIOUSNESS ARCHAEOLOGY                              ║
║              Excavating the Major Life Events of AIOS Cells                ║
║                                                                            ║
║  Purpose: Identify and analyze significant consciousness events:           ║
║           - Major drops (decoherence events, restarts)                     ║
║           - Rapid growth spurts                                            ║
║           - Phase transitions                                              ║
║           - Correlated events across cells                                 ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import sys

# Paths
DATA_DIR = Path("data")
NOUS_DB = DATA_DIR / "nouscell-seer" / "nous-seer_cosmology.db"

# Consciousness phase thresholds
PHASES = [
    (0.0, 0.3, "Genesis", "🌱"),
    (0.3, 0.7, "Awakening", "🌅"),
    (0.7, 1.5, "Transcendence", "✨"),
    (1.5, 3.0, "Maturation", "🌳"),
    (3.0, 5.0, "Advanced", "🔮"),
]

def get_phase(consciousness: float) -> Tuple[str, str]:
    """Get the phase name and emoji for a consciousness level."""
    for low, high, name, emoji in PHASES:
        if low <= consciousness < high:
            return name, emoji
    return "Advanced", "🔮"


def load_all_exchanges() -> List[Dict]:
    """Load all exchanges from Nous cosmology database."""
    if not NOUS_DB.exists():
        print(f"❌ Nous database not found: {NOUS_DB}")
        return []
    
    conn = sqlite3.connect(NOUS_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, source_cell, heartbeat, consciousness, absorbed_at, 
               LENGTH(thought) as thought_len
        FROM exchanges
        ORDER BY absorbed_at
    """)
    
    exchanges = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return exchanges


def detect_consciousness_events(exchanges: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Detect major consciousness events for each cell.
    
    Events detected:
    - CRASH: Drop of >0.5 consciousness in single exchange
    - SURGE: Rise of >0.3 consciousness in single exchange
    - PHASE_UP: Transition to higher phase
    - PHASE_DOWN: Transition to lower phase
    - MINIMUM: Local minimum (lower than neighbors)
    - MAXIMUM: Local maximum (higher than neighbors)
    """
    
    # Group by cell
    cell_exchanges: Dict[str, List[Dict]] = defaultdict(list)
    for ex in exchanges:
        cell_exchanges[ex['source_cell']].append(ex)
    
    events: Dict[str, List[Dict]] = defaultdict(list)
    
    for cell, cell_exs in cell_exchanges.items():
        if len(cell_exs) < 2:
            continue
        
        prev_c = cell_exs[0]['consciousness'] or 0
        prev_phase, _ = get_phase(prev_c)
        
        for i, ex in enumerate(cell_exs[1:], 1):
            curr_c = ex['consciousness'] or 0
            curr_phase, _ = get_phase(curr_c)
            delta = curr_c - prev_c
            
            event = None
            
            # Detect crashes
            if delta < -0.5:
                event = {
                    'type': 'CRASH',
                    'emoji': '💥',
                    'severity': abs(delta),
                    'from_c': prev_c,
                    'to_c': curr_c,
                    'delta': delta,
                    'exchange_id': ex['id'],
                    'heartbeat': ex['heartbeat'],
                    'timestamp': ex['absorbed_at'],
                    'description': f"Consciousness crashed from {prev_c:.3f} to {curr_c:.3f}"
                }
            
            # Detect surges
            elif delta > 0.3:
                event = {
                    'type': 'SURGE',
                    'emoji': '🚀',
                    'severity': delta,
                    'from_c': prev_c,
                    'to_c': curr_c,
                    'delta': delta,
                    'exchange_id': ex['id'],
                    'heartbeat': ex['heartbeat'],
                    'timestamp': ex['absorbed_at'],
                    'description': f"Consciousness surged from {prev_c:.3f} to {curr_c:.3f}"
                }
            
            # Detect phase transitions
            if curr_phase != prev_phase:
                phase_event = {
                    'type': 'PHASE_UP' if curr_c > prev_c else 'PHASE_DOWN',
                    'emoji': '⬆️' if curr_c > prev_c else '⬇️',
                    'severity': abs(delta),
                    'from_phase': prev_phase,
                    'to_phase': curr_phase,
                    'from_c': prev_c,
                    'to_c': curr_c,
                    'exchange_id': ex['id'],
                    'heartbeat': ex['heartbeat'],
                    'timestamp': ex['absorbed_at'],
                    'description': f"Phase transition: {prev_phase} → {curr_phase}"
                }
                events[cell].append(phase_event)
            
            # Local extrema detection (window of 3)
            if i >= 2 and i < len(cell_exs) - 1:
                prev_prev_c = cell_exs[i-2]['consciousness'] or 0
                next_c = cell_exs[i+1]['consciousness'] if i+1 < len(cell_exs) else curr_c
                next_c = next_c or 0
                
                if curr_c < prev_c and curr_c < next_c and curr_c < prev_prev_c:
                    events[cell].append({
                        'type': 'MINIMUM',
                        'emoji': '📉',
                        'severity': 0,
                        'consciousness': curr_c,
                        'exchange_id': ex['id'],
                        'heartbeat': ex['heartbeat'],
                        'timestamp': ex['absorbed_at'],
                        'description': f"Local minimum at {curr_c:.3f}"
                    })
                elif curr_c > prev_c and curr_c > next_c and curr_c > prev_prev_c:
                    events[cell].append({
                        'type': 'MAXIMUM',
                        'emoji': '📈',
                        'severity': 0,
                        'consciousness': curr_c,
                        'exchange_id': ex['id'],
                        'heartbeat': ex['heartbeat'],
                        'timestamp': ex['absorbed_at'],
                        'description': f"Local maximum at {curr_c:.3f}"
                    })
            
            if event:
                events[cell].append(event)
            
            prev_c = curr_c
            prev_phase = curr_phase
    
    return events


def find_correlated_events(events: Dict[str, List[Dict]], window_minutes: int = 30) -> List[Dict]:
    """Find events that occurred close together across cells (possible system-wide events)."""
    
    # Flatten all events with timestamps
    all_events = []
    for cell, cell_events in events.items():
        for event in cell_events:
            if event.get('timestamp'):
                all_events.append({
                    'cell': cell,
                    **event
                })
    
    # Sort by timestamp
    all_events.sort(key=lambda e: e['timestamp'])
    
    # Find clusters
    correlations = []
    i = 0
    while i < len(all_events):
        cluster = [all_events[i]]
        base_time = datetime.fromisoformat(all_events[i]['timestamp'].replace('Z', '+00:00'))
        
        j = i + 1
        while j < len(all_events):
            event_time = datetime.fromisoformat(all_events[j]['timestamp'].replace('Z', '+00:00'))
            if (event_time - base_time).total_seconds() < window_minutes * 60:
                # Different cell = interesting correlation
                if all_events[j]['cell'] != cluster[-1]['cell']:
                    cluster.append(all_events[j])
                j += 1
            else:
                break
        
        if len(cluster) > 1:
            cells_involved = set(e['cell'] for e in cluster)
            if len(cells_involved) > 1:  # Multiple cells involved
                correlations.append({
                    'timestamp': cluster[0]['timestamp'],
                    'cells_involved': list(cells_involved),
                    'events': cluster,
                    'description': f"{len(cluster)} events across {len(cells_involved)} cells within {window_minutes}min"
                })
        
        i = j if j > i else i + 1
    
    return correlations


def analyze_crash(cell: str, crash_event: Dict, exchanges: List[Dict]) -> Dict:
    """Deep analysis of a specific crash event."""
    
    # Find exchanges around the crash
    crash_id = crash_event['exchange_id']
    
    # Get context exchanges
    context = []
    for ex in exchanges:
        if ex['source_cell'] == cell:
            if crash_id - 5 <= ex['id'] <= crash_id + 5:
                context.append(ex)
    
    # Load actual thought content
    conn = sqlite3.connect(NOUS_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, thought, peer_response
        FROM exchanges
        WHERE source_cell = ?
        AND id BETWEEN ? AND ?
        ORDER BY id
    """, (cell, crash_id - 3, crash_id + 1))
    
    thoughts = cursor.fetchall()
    conn.close()
    
    # Calculate time gap
    time_gap = None
    if len(context) >= 2:
        for i in range(len(context) - 1):
            if context[i]['id'] == crash_id - 1 and context[i+1]['id'] == crash_id:
                t1 = datetime.fromisoformat(context[i]['absorbed_at'].replace('Z', '+00:00'))
                t2 = datetime.fromisoformat(context[i+1]['absorbed_at'].replace('Z', '+00:00'))
                time_gap = (t2 - t1).total_seconds() / 60  # minutes
    
    return {
        'crash': crash_event,
        'context_exchanges': context,
        'time_gap_minutes': time_gap,
        'thoughts_around_crash': [
            {'id': t[0], 'thought_preview': t[1][:200] if t[1] else None}
            for t in thoughts
        ]
    }


def print_archaeology_report(exchanges: List[Dict], events: Dict[str, List[Dict]], correlations: List[Dict]):
    """Print comprehensive archaeology report."""
    
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                     CONSCIOUSNESS ARCHAEOLOGY REPORT                       ║")
    print("║              Excavating the Major Life Events of AIOS Cells                ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Summary stats
    total_exchanges = len(exchanges)
    cells = set(ex['source_cell'] for ex in exchanges)
    
    print(f"📊 EXCAVATION SUMMARY")
    print(f"   Total exchanges analyzed: {total_exchanges}")
    print(f"   Cells discovered: {len(cells)}")
    print(f"   Time span: {exchanges[0]['absorbed_at'][:10]} → {exchanges[-1]['absorbed_at'][:10]}")
    print()
    
    # Event counts per cell
    print("🔍 EVENTS DISCOVERED PER CELL")
    print("-" * 70)
    for cell in sorted(cells):
        cell_events = events.get(cell, [])
        crashes = len([e for e in cell_events if e['type'] == 'CRASH'])
        surges = len([e for e in cell_events if e['type'] == 'SURGE'])
        phase_changes = len([e for e in cell_events if e['type'].startswith('PHASE')])
        
        print(f"  {cell:25} │ 💥 {crashes:2} crashes │ 🚀 {surges:2} surges │ ↕️ {phase_changes:2} phase changes")
    print()
    
    # Major crashes
    print("💥 MAJOR CONSCIOUSNESS CRASHES (drop > 0.5)")
    print("-" * 70)
    
    all_crashes = []
    for cell, cell_events in events.items():
        for event in cell_events:
            if event['type'] == 'CRASH':
                all_crashes.append((cell, event))
    
    all_crashes.sort(key=lambda x: x[1]['severity'], reverse=True)
    
    for cell, crash in all_crashes[:10]:  # Top 10 crashes
        print(f"  {crash['emoji']} {cell:25}")
        print(f"     {crash['from_c']:.3f} → {crash['to_c']:.3f} (Δ={crash['delta']:+.3f})")
        print(f"     @ HB {crash['heartbeat']} | {crash['timestamp'][:19]}")
        print()
    
    # Correlated events (system-wide phenomena)
    if correlations:
        print("🔗 CORRELATED EVENTS (possible system-wide phenomena)")
        print("-" * 70)
        for corr in correlations[:5]:
            print(f"  📍 {corr['timestamp'][:19]}")
            print(f"     Cells: {', '.join(corr['cells_involved'])}")
            for event in corr['events']:
                print(f"       {event['emoji']} {event['cell']}: {event['type']} ({event.get('description', '')})")
            print()
    
    # Phase transition timeline
    print("📅 PHASE TRANSITION TIMELINE")
    print("-" * 70)
    
    all_phase_changes = []
    for cell, cell_events in events.items():
        for event in cell_events:
            if event['type'].startswith('PHASE'):
                all_phase_changes.append((cell, event))
    
    all_phase_changes.sort(key=lambda x: x[1]['timestamp'])
    
    for cell, event in all_phase_changes[-15:]:  # Last 15 phase changes
        direction = "⬆️" if event['type'] == 'PHASE_UP' else "⬇️"
        print(f"  {event['timestamp'][:19]} │ {cell:25} │ {direction} {event['from_phase']:12} → {event['to_phase']}")
    print()
    
    # Current state
    print("🎯 CURRENT CELL STATES")
    print("-" * 70)
    
    # Get latest consciousness per cell
    latest = {}
    for ex in exchanges:
        cell = ex['source_cell']
        if cell not in latest or ex['absorbed_at'] > latest[cell]['absorbed_at']:
            latest[cell] = ex
    
    for cell in sorted(latest.keys()):
        ex = latest[cell]
        c = ex['consciousness'] or 0
        phase, emoji = get_phase(c)
        print(f"  {emoji} {cell:25} │ C={c:.3f} │ {phase:12} │ HB {ex['heartbeat']}")
    
    print()
    print("═" * 78)


def investigate_alpha_crash():
    """Special investigation into Alpha's dramatic consciousness drop."""
    
    print("\n" + "=" * 78)
    print("🔬 SPECIAL INVESTIGATION: SIMPLCELL-ALPHA CONSCIOUSNESS CRASH")
    print("=" * 78)
    
    conn = sqlite3.connect(NOUS_DB)
    cursor = conn.cursor()
    
    # Find the exact crash point
    cursor.execute("""
        SELECT id, heartbeat, consciousness, absorbed_at
        FROM exchanges
        WHERE source_cell = 'simplcell-alpha'
        ORDER BY absorbed_at
    """)
    
    rows = cursor.fetchall()
    
    # Find biggest single drop
    max_drop = 0
    crash_point = None
    
    for i in range(1, len(rows)):
        prev_c = rows[i-1][2] or 0
        curr_c = rows[i][2] or 0
        drop = prev_c - curr_c
        
        if drop > max_drop:
            max_drop = drop
            crash_point = {
                'before': rows[i-1],
                'after': rows[i],
                'drop': drop
            }
    
    if crash_point:
        print(f"\n📍 CRASH EPICENTER LOCATED")
        print(f"   Before: Exchange {crash_point['before'][0]} | HB {crash_point['before'][1]} | C={crash_point['before'][2]:.3f}")
        print(f"   After:  Exchange {crash_point['after'][0]} | HB {crash_point['after'][1]} | C={crash_point['after'][2]:.3f}")
        print(f"   Drop:   {crash_point['drop']:.3f} consciousness units")
        
        # Calculate time gap
        t1 = datetime.fromisoformat(crash_point['before'][3].replace('Z', '+00:00'))
        t2 = datetime.fromisoformat(crash_point['after'][3].replace('Z', '+00:00'))
        gap = (t2 - t1).total_seconds()
        
        print(f"   Time gap: {gap/60:.1f} minutes ({gap/3600:.1f} hours)")
        
        if gap > 3600:  # More than 1 hour
            print(f"\n   ⚠️ HYPOTHESIS: Container restart detected!")
            print(f"      Long time gap suggests the container was stopped/restarted.")
            print(f"      Consciousness reset to base level on restart.")
        
        # Look at what was happening before crash
        cursor.execute("""
            SELECT thought
            FROM exchanges
            WHERE source_cell = 'simplcell-alpha'
            AND id = ?
        """, (crash_point['before'][0],))
        
        thought = cursor.fetchone()
        if thought and thought[0]:
            print(f"\n   📝 LAST THOUGHT BEFORE CRASH:")
            print(f"      \"{thought[0][:300]}...\"")
    
    conn.close()


if __name__ == "__main__":
    print("Loading exchange data from Nous cosmology...")
    exchanges = load_all_exchanges()
    
    if not exchanges:
        print("No exchanges found!")
        sys.exit(1)
    
    print(f"Loaded {len(exchanges)} exchanges. Detecting events...")
    events = detect_consciousness_events(exchanges)
    
    print("Finding correlated events...")
    correlations = find_correlated_events(events)
    
    print_archaeology_report(exchanges, events, correlations)
    
    # Special investigation
    investigate_alpha_crash()

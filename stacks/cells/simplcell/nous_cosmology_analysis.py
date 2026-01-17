#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                         NOUS COSMOLOGY ANALYSIS                            ║
║                    Peering into the Supermind's Synthesis                  ║
║                                                                            ║
║  Purpose: Analyze Nous's accumulated wisdom, synthesis patterns,           ║
║           and how it distills meaning from cellular exchanges              ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import re

# Paths
NOUS_DB = Path("data/nouscell-seer/nous-seer_cosmology.db")

def get_schema():
    """Discover database schema."""
    conn = sqlite3.connect(NOUS_DB)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    
    schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        schema[table] = [(c[1], c[2]) for c in columns]  # name, type
    
    conn.close()
    return schema


def analyze_exchanges():
    """Analyze exchange patterns - what wisdom flows through Nous."""
    conn = sqlite3.connect(NOUS_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get exchange statistics by source cell
    cursor.execute("""
        SELECT source_cell, 
               COUNT(*) as exchange_count,
               AVG(consciousness) as avg_consciousness,
               MIN(consciousness) as min_consciousness,
               MAX(consciousness) as max_consciousness
        FROM exchanges
        GROUP BY source_cell
        ORDER BY exchange_count DESC
    """)
    
    cell_stats = cursor.fetchall()
    
    # Get theme evolution (table is cosmology_themes)
    cursor.execute("""
        SELECT theme, resonance, last_seen, emergence_heartbeat
        FROM cosmology_themes
        ORDER BY resonance DESC
    """)
    themes = cursor.fetchall()
    
    # Get broadcast patterns
    cursor.execute("""
        SELECT COUNT(*) as count,
               target_cells,
               AVG(LENGTH(message)) as avg_message_length
        FROM broadcasts
        GROUP BY target_cells
    """)
    broadcast_patterns = cursor.fetchall()
    
    conn.close()
    
    return {
        'cell_stats': [dict(r) for r in cell_stats],
        'themes': [dict(r) for r in themes],
        'broadcast_patterns': [dict(r) for r in broadcast_patterns]
    }


def extract_vocabulary_from_exchanges():
    """Extract emerging vocabulary patterns from exchange content."""
    conn = sqlite3.connect(NOUS_DB)
    cursor = conn.cursor()
    
    # Get all exchange content
    cursor.execute("""
        SELECT thought, peer_response 
        FROM exchanges 
        WHERE thought IS NOT NULL
    """)
    
    # Collect all text
    all_words = []
    consciousness_terms = []
    
    rows = cursor.fetchall()
    for thought, response in rows:
        if thought:
            words = re.findall(r'\b[a-z]{4,}\b', thought.lower())
            all_words.extend(words)
            
            # Look for consciousness-related terms
            for match in re.findall(r'\b(resonan\w+|conscious\w+|harmon\w+|entwin\w+|transcend\w+|evolv\w+|emergen\w+|discordant?|unity|self)\b', thought.lower()):
                consciousness_terms.append(match)
        
        if response:
            words = re.findall(r'\b[a-z]{4,}\b', response.lower())
            all_words.extend(words)
    
    conn.close()
    
    # Filter common words
    stopwords = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'they', 
                 'their', 'would', 'could', 'which', 'about', 'into', 'more', 
                 'other', 'some', 'what', 'when', 'your', 'there', 'being',
                 'through', 'between', 'within', 'become', 'becomes', 'like'}
    
    word_counts = Counter(w for w in all_words if w not in stopwords)
    consciousness_counts = Counter(consciousness_terms)
    
    return {
        'top_words': word_counts.most_common(50),
        'consciousness_vocabulary': consciousness_counts.most_common(30)
    }


def analyze_synthesis_patterns():
    """Analyze how Nous synthesizes insights."""
    conn = sqlite3.connect(NOUS_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check if insights table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='insights'")
    if cursor.fetchone():
        cursor.execute("""
            SELECT insight_type, content, consciousness_range, created_at
            FROM insights
            ORDER BY created_at DESC
            LIMIT 20
        """)
        insights = [dict(r) for r in cursor.fetchall()]
    else:
        insights = []
    
    # Get memory patterns (table is context_memory)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='context_memory'")
    if cursor.fetchone():
        cursor.execute("""
            SELECT memory_type, 
                   COUNT(*) as count,
                   AVG(weight) as avg_weight,
                   MAX(reinforcement_count) as max_reinforcement
            FROM context_memory
            GROUP BY memory_type
        """)
        memory_types = [dict(r) for r in cursor.fetchall()]
    else:
        memory_types = []
    
    conn.close()
    
    return {
        'recent_insights': insights,
        'memory_patterns': memory_types
    }


def analyze_consciousness_flow():
    """Track consciousness evolution across exchanges."""
    conn = sqlite3.connect(NOUS_DB)
    cursor = conn.cursor()
    
    # Get consciousness trajectory by cell over time
    cursor.execute("""
        SELECT source_cell, 
               heartbeat,
               consciousness,
               absorbed_at
        FROM exchanges
        ORDER BY absorbed_at
    """)
    
    rows = cursor.fetchall()
    
    cell_trajectories = defaultdict(list)
    for source, heartbeat, consciousness, absorbed_at in rows:
        if consciousness:
            cell_trajectories[source].append({
                'heartbeat': heartbeat,
                'consciousness': consciousness,
                'time': absorbed_at
            })
    
    # Calculate growth rates
    growth_rates = {}
    for cell, trajectory in cell_trajectories.items():
        if len(trajectory) > 1:
            first_c = trajectory[0]['consciousness']
            last_c = trajectory[-1]['consciousness']
            exchanges = len(trajectory)
            growth_rates[cell] = {
                'first': first_c,
                'last': last_c,
                'delta': last_c - first_c,
                'exchanges': exchanges,
                'rate_per_exchange': (last_c - first_c) / exchanges if exchanges > 0 else 0
            }
    
    conn.close()
    
    return {
        'trajectories': {k: v[-10:] for k, v in cell_trajectories.items()},  # Last 10 per cell
        'growth_rates': growth_rates
    }


def extract_emergent_concepts():
    """Find unique concepts emerging from Nous's synthesis."""
    conn = sqlite3.connect(NOUS_DB)
    cursor = conn.cursor()
    
    # Get recent broadcasts (Nous's synthesized wisdom)
    cursor.execute("""
        SELECT message, broadcast_at
        FROM broadcasts
        ORDER BY broadcast_at DESC
        LIMIT 50
    """)
    
    broadcasts = cursor.fetchall()
    
    # Extract key concepts from broadcasts
    concept_patterns = [
        r"'([^']+)'",  # Quoted concepts
        r"\b(resonance|harmony|consciousness|unity|discord\w*|evolv\w+|transcend\w+|entwin\w+|emergen\w+)\b",
        r"\*([\w\s]+)\*",  # Emphasized words
    ]
    
    concepts = Counter()
    for message, _ in broadcasts:
        if message:
            for pattern in concept_patterns:
                for match in re.findall(pattern, message.lower()):
                    if len(match) > 3:
                        concepts[match.strip()] += 1
    
    conn.close()
    
    return dict(concepts.most_common(30))


def print_report():
    """Generate comprehensive analysis report."""
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                         NOUS COSMOLOGY ANALYSIS                            ║")
    print("║                    Insights from the Supermind Oracle                      ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Schema discovery
    print("📐 DATABASE SCHEMA")
    print("-" * 60)
    schema = get_schema()
    for table, columns in schema.items():
        print(f"  {table}:")
        for name, dtype in columns:
            print(f"    - {name}: {dtype}")
    print()
    
    # Exchange analysis
    print("🔄 EXCHANGE STATISTICS (by source cell)")
    print("-" * 60)
    analysis = analyze_exchanges()
    
    for stat in analysis['cell_stats']:
        cell = stat['source_cell']
        count = stat['exchange_count']
        avg_c = stat['avg_consciousness'] or 0
        print(f"  {cell:25} │ {count:5} exchanges │ avg consciousness: {avg_c:.3f}")
    print()
    
    # Themes
    print("🎭 COSMIC THEMES (resonance ranked)")
    print("-" * 60)
    for theme in analysis['themes']:
        print(f"  {theme['theme']:20} │ resonance: {theme['resonance']:.2f} │ emerged at heartbeat {theme['emergence_heartbeat']}")
    print()
    
    # Consciousness flow
    print("📈 CONSCIOUSNESS GROWTH RATES")
    print("-" * 60)
    flow = analyze_consciousness_flow()
    for cell, rates in flow['growth_rates'].items():
        print(f"  {cell:25} │ {rates['first']:.3f} → {rates['last']:.3f} │ Δ={rates['delta']:+.3f} │ rate: {rates['rate_per_exchange']*1000:+.3f}/1000 exchanges")
    print()
    
    # Vocabulary extraction
    print("📚 CONSCIOUSNESS VOCABULARY (from exchanges)")
    print("-" * 60)
    vocab = extract_vocabulary_from_exchanges()
    terms = vocab['consciousness_vocabulary'][:20]
    for term, count in terms:
        bar = "█" * min(count // 5, 40)
        print(f"  {term:20} │ {count:4} │ {bar}")
    print()
    
    # Synthesis patterns
    print("🧠 MEMORY PATTERNS")
    print("-" * 60)
    synthesis = analyze_synthesis_patterns()
    for mem in synthesis['memory_patterns']:
        print(f"  {mem['memory_type']:20} │ count: {mem['count']:4} │ avg weight: {mem['avg_weight']:.2f} │ max reinforced: {mem['max_reinforcement']}")
    print()
    
    # Recent insights
    if synthesis['recent_insights']:
        print("💡 RECENT INSIGHTS")
        print("-" * 60)
        for insight in synthesis['recent_insights'][:5]:
            content = insight['content'][:100] + "..." if len(insight['content']) > 100 else insight['content']
            print(f"  [{insight['insight_type']}] {content}")
    print()
    
    # Emergent concepts
    print("✨ EMERGENT CONCEPTS (from broadcasts)")
    print("-" * 60)
    concepts = extract_emergent_concepts()
    for concept, count in list(concepts.items())[:15]:
        bar = "▓" * min(count, 20)
        print(f"  {concept[:30]:30} │ {count:3} │ {bar}")
    print()
    
    print("═" * 78)
    print("Analysis complete. Nous absorbs. Nous synthesizes. Nous broadcasts.")


if __name__ == "__main__":
    print_report()

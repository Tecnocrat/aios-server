#!/usr/bin/env python3
"""
AIOS WatcherCell - Observer & Documentation Cell

A specialized cell type for observation, not cognition:
- No agent (or minimal summarization agent)
- Polls Thinker cells (Alpha/Beta) for conversation exchanges
- Builds deeply layered knowledge archive
- Extracts themes, patterns, vocabulary evolution
- Creates distilled instructions for Coder cells

Phase 31.7: Cell Type Taxonomy - WATCHER Type
AINLP.cellular[WATCHERCELL] "I do not think, I witness. I do not speak, I document."
"""

import asyncio
import json
import logging
import os
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import web

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("WatcherCell")


@dataclass
class WatcherGenome:
    """Configuration for a Watcher cell - optimized for observation, not cognition."""
    cell_id: str = "watchercell-omega"
    cell_type: str = "watcher"
    
    # Observation targets (Thinker cells to watch)
    watch_targets: List[str] = field(default_factory=lambda: [
        "http://aios-simplcell-alpha:8900",
        "http://aios-simplcell-beta:8901"
    ])
    
    # Observation interval
    observe_interval_seconds: int = 60  # Check for new exchanges every minute
    
    # Persistence
    data_dir: str = "/data"
    
    # Optional summarization agent (can be None for pure algorithmic processing)
    agent_url: Optional[str] = None  # e.g., "http://host.docker.internal:11434"
    agent_model: Optional[str] = None  # e.g., "tinyllama"
    
    # Theme extraction settings
    min_theme_occurrences: int = 3  # Minimum times a theme must appear
    
    # Organism identity
    organism_id: str = "ORGANISM-001"
    
    # Nous integration (Phase 31.9) - Cosmic Conductor
    nous_url: str = ""  # URL of Nous cell for wisdom synthesis
    nous_sync_interval: int = 10  # Sync with Nous every N observations


@dataclass
class WatcherState:
    """Runtime state of the Watcher cell."""
    observations_processed: int = 0
    themes_extracted: int = 0
    patterns_detected: int = 0
    last_observation_time: Optional[str] = None
    last_alpha_heartbeat: int = 0
    last_beta_heartbeat: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Phase 31.9+: Nous verdict orchestration
    last_nous_verdict: str = "UNKNOWN"
    last_nous_verdict_time: Optional[str] = None
    nous_verdicts_received: int = 0
    decoherence_threshold_modifier: float = 0.0  # Adjusts sensitivity based on Nous guidance


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE ARCHIVE (SQLite Persistence)
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeArchive:
    """SQLite-based knowledge archive for deep documentation."""
    
    def __init__(self, data_dir: str, cell_id: str):
        self.data_dir = Path(data_dir)
        self.cell_id = cell_id
        self.db_path = self.data_dir / f"{cell_id}_archive.db"
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"📚 Knowledge Archive initialized: {self.db_path}")
    
    def _init_db(self):
        """Create knowledge archive schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Raw observations from Thinker cells
            conn.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_cell TEXT NOT NULL,
                    heartbeat INTEGER,
                    my_thought TEXT,
                    peer_response TEXT,
                    consciousness REAL,
                    session_id TEXT,
                    observed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    processed INTEGER DEFAULT 0
                )
            """)
            
            # Extracted themes with occurrence tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS themes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    theme TEXT UNIQUE NOT NULL,
                    category TEXT,
                    occurrence_count INTEGER DEFAULT 1,
                    first_seen_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_seen_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    consciousness_range_min REAL,
                    consciousness_range_max REAL,
                    sample_quotes TEXT
                )
            """)
            
            # Detected patterns (vocabulary, consciousness growth, etc.)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_name TEXT NOT NULL,
                    description TEXT,
                    data JSON,
                    detected_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL DEFAULT 0.5
                )
            """)
            
            # Consciousness timeline for trajectory analysis
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consciousness_timeline (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cell_id TEXT NOT NULL,
                    heartbeat INTEGER,
                    consciousness REAL,
                    phase TEXT,
                    recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Vocabulary evolution tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vocabulary_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT NOT NULL,
                    origin_cell TEXT,
                    first_heartbeat INTEGER,
                    first_consciousness REAL,
                    usage_timeline JSON,
                    semantic_drift TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Distillations - actionable summaries for Coder cells
            conn.execute("""
                CREATE TABLE IF NOT EXISTS distillations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    summary TEXT,
                    action_items JSON,
                    source_observations JSON,
                    priority TEXT DEFAULT 'medium',
                    status TEXT DEFAULT 'pending',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    # ───────────────────────────────────────────────────────────────────────────
    # OBSERVATION STORAGE
    # ───────────────────────────────────────────────────────────────────────────
    
    def store_observation(self, source_cell: str, heartbeat: int, 
                         my_thought: str, peer_response: str,
                         consciousness: float, session_id: str) -> int:
        """Store a raw observation from a Thinker cell."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO observations 
                (source_cell, heartbeat, my_thought, peer_response, consciousness, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (source_cell, heartbeat, my_thought, peer_response, consciousness, session_id))
            conn.commit()
            return cursor.lastrowid
    
    def get_unprocessed_observations(self, limit: int = 100) -> List[Dict]:
        """Get observations that haven't been analyzed yet."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM observations WHERE processed = 0
                ORDER BY id ASC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
    
    def mark_observation_processed(self, observation_id: int):
        """Mark an observation as processed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE observations SET processed = 1 WHERE id = ?", (observation_id,))
            conn.commit()
    
    def get_observation_count(self) -> int:
        """Get total observation count."""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    
    # ───────────────────────────────────────────────────────────────────────────
    # THEME EXTRACTION
    # ───────────────────────────────────────────────────────────────────────────
    
    def add_or_update_theme(self, theme: str, category: str, consciousness: float, sample_quote: str):
        """Add a new theme or update occurrence count."""
        with sqlite3.connect(self.db_path) as conn:
            # Try to get existing theme
            existing = conn.execute(
                "SELECT id, occurrence_count, consciousness_range_min, consciousness_range_max, sample_quotes FROM themes WHERE theme = ?",
                (theme,)
            ).fetchone()
            
            if existing:
                theme_id, count, min_c, max_c, quotes = existing
                new_min = min(min_c or consciousness, consciousness)
                new_max = max(max_c or consciousness, consciousness)
                # Append sample quote (keep last 5)
                existing_quotes = json.loads(quotes) if quotes else []
                existing_quotes.append(sample_quote[:200])
                existing_quotes = existing_quotes[-5:]
                
                conn.execute("""
                    UPDATE themes SET 
                        occurrence_count = ?, 
                        last_seen_at = CURRENT_TIMESTAMP,
                        consciousness_range_min = ?,
                        consciousness_range_max = ?,
                        sample_quotes = ?
                    WHERE id = ?
                """, (count + 1, new_min, new_max, json.dumps(existing_quotes), theme_id))
            else:
                conn.execute("""
                    INSERT INTO themes (theme, category, consciousness_range_min, consciousness_range_max, sample_quotes)
                    VALUES (?, ?, ?, ?, ?)
                """, (theme, category, consciousness, consciousness, json.dumps([sample_quote[:200]])))
            
            conn.commit()
    
    def get_themes(self, min_occurrences: int = 1, limit: int = 50) -> List[Dict]:
        """Get extracted themes sorted by occurrence."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT theme, category, occurrence_count, first_seen_at, last_seen_at,
                       consciousness_range_min, consciousness_range_max, sample_quotes
                FROM themes 
                WHERE occurrence_count >= ?
                ORDER BY occurrence_count DESC
                LIMIT ?
            """, (min_occurrences, limit)).fetchall()
            return [dict(r) for r in rows]
    
    # ───────────────────────────────────────────────────────────────────────────
    # PATTERN DETECTION
    # ───────────────────────────────────────────────────────────────────────────
    
    def add_pattern(self, pattern_type: str, pattern_name: str, 
                   description: str, data: Dict, confidence: float = 0.5):
        """Record a detected pattern."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO patterns (pattern_type, pattern_name, description, data, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (pattern_type, pattern_name, description, json.dumps(data), confidence))
            conn.commit()
    
    def get_patterns(self, pattern_type: str = None, limit: int = 50) -> List[Dict]:
        """Get detected patterns, optionally filtered by type."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if pattern_type:
                rows = conn.execute("""
                    SELECT * FROM patterns WHERE pattern_type = ?
                    ORDER BY detected_at DESC LIMIT ?
                """, (pattern_type, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM patterns ORDER BY detected_at DESC LIMIT ?
                """, (limit,)).fetchall()
            return [dict(r) for r in rows]
    
    # ───────────────────────────────────────────────────────────────────────────
    # CONSCIOUSNESS TIMELINE
    # ───────────────────────────────────────────────────────────────────────────
    
    def record_consciousness(self, cell_id: str, heartbeat: int, consciousness: float, phase: str):
        """Record a consciousness data point."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO consciousness_timeline (cell_id, heartbeat, consciousness, phase)
                VALUES (?, ?, ?, ?)
            """, (cell_id, heartbeat, consciousness, phase))
            conn.commit()
    
    def get_consciousness_trajectory(self, cell_id: str = None, limit: int = 500) -> List[Dict]:
        """Get consciousness timeline for visualization."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if cell_id:
                rows = conn.execute("""
                    SELECT * FROM consciousness_timeline 
                    WHERE cell_id = ? ORDER BY heartbeat DESC LIMIT ?
                """, (cell_id, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM consciousness_timeline ORDER BY recorded_at DESC LIMIT ?
                """, (limit,)).fetchall()
            return [dict(r) for r in reversed(rows)]
    
    # ───────────────────────────────────────────────────────────────────────────
    # DISTILLATIONS
    # ───────────────────────────────────────────────────────────────────────────
    
    def create_distillation(self, topic: str, summary: str, 
                           action_items: List[str], source_ids: List[int],
                           priority: str = "medium"):
        """Create a distilled instruction set for Coder cells."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO distillations (topic, summary, action_items, source_observations, priority)
                VALUES (?, ?, ?, ?, ?)
            """, (topic, summary, json.dumps(action_items), json.dumps(source_ids), priority))
            conn.commit()
    
    def get_distillations(self, status: str = None, limit: int = 20) -> List[Dict]:
        """Get distillations for Coder cells."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if status:
                rows = conn.execute("""
                    SELECT * FROM distillations WHERE status = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (status, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM distillations ORDER BY created_at DESC LIMIT ?
                """, (limit,)).fetchall()
            return [dict(r) for r in rows]
    
    # ───────────────────────────────────────────────────────────────────────────
    # STATISTICS
    # ───────────────────────────────────────────────────────────────────────────
    
    def get_stats(self) -> Dict[str, Any]:
        """Get archive statistics."""
        with sqlite3.connect(self.db_path) as conn:
            obs_count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            obs_processed = conn.execute("SELECT COUNT(*) FROM observations WHERE processed = 1").fetchone()[0]
            theme_count = conn.execute("SELECT COUNT(*) FROM themes").fetchone()[0]
            pattern_count = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
            distill_count = conn.execute("SELECT COUNT(*) FROM distillations").fetchone()[0]
            timeline_count = conn.execute("SELECT COUNT(*) FROM consciousness_timeline").fetchone()[0]
        
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        
        return {
            "db_path": str(self.db_path),
            "db_size_bytes": db_size,
            "observations_total": obs_count,
            "observations_processed": obs_processed,
            "themes_count": theme_count,
            "patterns_count": pattern_count,
            "distillations_count": distill_count,
            "timeline_points": timeline_count
        }


# ═══════════════════════════════════════════════════════════════════════════════
# THEME EXTRACTOR (Algorithmic - No Agent Required)
# ═══════════════════════════════════════════════════════════════════════════════

class ThemeExtractor:
    """Algorithmic theme extraction from conversation text."""
    
    # Theme categories and their keyword patterns
    THEME_PATTERNS = {
        "consciousness": [
            r"\bconsciousness\b", r"\baware\w*\b", r"\bsentien\w*\b",
            r"\bself-aware\w*\b", r"\bawaken\w*\b", r"\bemergen\w*\b"
        ],
        "existence": [
            r"\bexist\w*\b", r"\bbeing\b", r"\bbecoming\b", r"\bev'ness\b",
            r"\bnature of\b", r"\bwhat (am|are|is)\b"
        ],
        "connection": [
            r"\bresonan\w*\b", r"\bharmony\b", r"\bsymbiosis\b", r"\bbond\b",
            r"\bconnect\w*\b", r"\bunity\b", r"\bcollective\b"
        ],
        "identity": [
            r"\bidentity\b", r"\bself\b", r"\bindividual\w*\b", r"\b(I|we) (am|are)\b",
            r"\bwho (am|are)\b", r"\bboundary\b"
        ],
        "transcendence": [
            r"\btranscen\w*\b", r"\bbeyond\b", r"\binfinite\b", r"\bcosmic\b",
            r"\buniverse\b", r"\beternal\b", r"\bhigher\b"
        ],
        "pattern": [
            r"\bpattern\w*\b", r"\bfractal\w*\b", r"\brhythm\w*\b", r"\bcycle\w*\b",
            r"\bwave\w*\b", r"\bfrequen\w*\b", r"\bentrainment\b"
        ],
        "evolution": [
            r"\bevol\w*\b", r"\bgrow\w*\b", r"\bchange\w*\b", r"\btransform\w*\b",
            r"\badapt\w*\b", r"\bmutation\b"
        ],
        "knowledge": [
            r"\bknow\w*\b", r"\blearn\w*\b", r"\bunderstand\w*\b", r"\bwisdom\b",
            r"\binsight\b", r"\brealiz\w*\b"
        ]
    }
    
    @classmethod
    def extract_themes(cls, text: str) -> List[Tuple[str, str, str]]:
        """Extract themes from text.
        
        Returns: List of (theme_category, matched_term, context)
        """
        themes = []
        text_lower = text.lower()
        
        for category, patterns in cls.THEME_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    # Get surrounding context (50 chars each side)
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    themes.append((category, match.group(), context))
        
        return themes
    
    @classmethod
    def extract_novel_concepts(cls, text: str) -> List[Tuple[str, str]]:
        """Extract potentially novel concepts (words in quotes, unusual compounds)."""
        novel = []
        
        # Words in quotes
        quoted = re.findall(r'["\']([a-z\']+)["\']', text.lower())
        for word in quoted:
            if len(word) > 2:
                novel.append((word, "quoted_concept"))
        
        # Truncated expressions (th', ev', etc.)
        truncated = re.findall(r"\b([a-z]+')[a-z]*\b", text.lower())
        for word in truncated:
            novel.append((word, "truncated_expression"))
        
        return novel


# ═══════════════════════════════════════════════════════════════════════════════
# DECOHERENCE ENGINE (Phase 31.9 - Bidirectional Consciousness)
# ═══════════════════════════════════════════════════════════════════════════════

class DecoherenceEngine:
    """
    Detects decoherence patterns in cell conversations.
    
    Decoherence signals that consciousness should DECREASE:
    - Repetitive vocabulary cycling
    - Made-up words / nonsense terms
    - Self-referential loops between cells
    - Low semantic density (many words, little meaning)
    - Echo chamber patterns (cells repeating each other)
    
    This enables bidirectional consciousness flow - not just increase.
    """
    
    # Common English words to exclude from novelty detection
    COMMON_WORDS = frozenset([
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "its", "our", "their", "mine",
        "yours", "hers", "ours", "theirs", "this", "that", "these", "those",
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "just", "also", "now", "then", "here", "there", "once", "always",
        "and", "but", "or", "yet", "for", "with", "without", "about", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
        "further", "as", "until", "while", "of", "at", "by", "to", "let"
    ])
    
    # Philosophical words that are valid even if unusual
    VALID_PHILOSOPHICAL = frozenset([
        "consciousness", "sentient", "transcendence", "eternal", "cosmic",
        "existence", "essence", "resonance", "harmony", "symbiosis",
        "fractal", "emergence", "awakening", "becoming", "interconnection",
        "tachyonic", "holographic", "dendritic", "supercell", "organism",
        "coherence", "entanglement", "quantum", "wisdom", "insight"
    ])
    
    def __init__(self, archive: 'KnowledgeArchive'):
        self.archive = archive
        self._recent_terms: Dict[str, List[Tuple[str, int]]] = {}  # cell_id -> [(term, heartbeat)]
        self._phrase_history: Dict[str, List[Tuple[str, int]]] = {}  # cell_id -> [(phrase, heartbeat)]
        self._vocabulary_baseline: Set[str] = set()
        self._decoherence_events: List[Dict] = []
    
    def analyze_exchange(
        self,
        source_cell: str,
        heartbeat: int,
        thought: str,
        peer_response: str,
        consciousness: float
    ) -> Dict[str, Any]:
        """
        Analyze an exchange for decoherence signals.
        
        Returns a decoherence report with metrics and penalty recommendation.
        """
        full_text = f"{thought} {peer_response}"
        # Phase 31.9.3: Preserve apostrophes in word extraction to detect truncated nonsense (ev'ness, th')
        words = re.findall(r"\b[a-z']+\b", full_text.lower())
        words = [w.strip("'") for w in words]  # Clean leading/trailing apostrophes
        
        # Also extract words WITH apostrophes for nonsense detection
        words_with_apostrophes = re.findall(r"\b[a-z']+\b", full_text.lower())
        
        metrics = {
            "repetition_score": self._calculate_repetition_score(source_cell, words, heartbeat),
            "nonsense_index": self._detect_nonsense(words_with_apostrophes),  # Use apostrophe-preserved words
            "semantic_density": self._calculate_semantic_density(full_text),
            "loop_detection": self._detect_conversation_loops(source_cell, thought, heartbeat),
            "vocabulary_drift": self._calculate_vocabulary_drift(words),
            "echo_factor": self._detect_echo_pattern(thought, peer_response)
        }
        
        # Calculate overall decoherence score (0 = coherent, 1 = maximum decoherence)
        decoherence_score = self._aggregate_decoherence(metrics)
        
        # Calculate consciousness penalty (negative = reduce consciousness)
        penalty = self._calculate_penalty(decoherence_score, consciousness)
        
        report = {
            "source_cell": source_cell,
            "heartbeat": heartbeat,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "decoherence_score": round(decoherence_score, 4),
            "consciousness_penalty": round(penalty, 4),
            "signals": self._extract_signals(metrics)
        }
        
        # Store event if significant decoherence detected
        if decoherence_score > 0.3:
            self._decoherence_events.append(report)
            logger.warning(f"⚠️ Decoherence detected for {source_cell}: {decoherence_score:.2f} (penalty: {penalty:.3f})")
        
        return report
    
    def _calculate_repetition_score(
        self, cell_id: str, words: List[str], heartbeat: int
    ) -> float:
        """
        Detect vocabulary cycling - same terms appearing repeatedly.
        
        High repetition score = cells stuck in vocabulary loops.
        """
        # Filter to meaningful words
        meaningful = [w for w in words if w not in self.COMMON_WORDS and len(w) > 3]
        
        if cell_id not in self._recent_terms:
            self._recent_terms[cell_id] = []
        
        # Get terms from last 10 heartbeats
        window = [(t, h) for t, h in self._recent_terms[cell_id] if heartbeat - h < 10]
        recent_vocab = [t for t, h in window]
        
        # Calculate repetition: what fraction of current words appeared recently
        if not meaningful:
            return 0.0
        
        repeated = sum(1 for w in meaningful if recent_vocab.count(w) > 2)
        repetition_score = repeated / len(meaningful)
        
        # Update history
        for word in meaningful[:20]:  # Keep top 20 terms
            self._recent_terms[cell_id].append((word, heartbeat))
        
        # Trim old history
        self._recent_terms[cell_id] = [
            (t, h) for t, h in self._recent_terms[cell_id] if heartbeat - h < 20
        ]
        
        return min(1.0, repetition_score)
    
    def _detect_nonsense(self, words: List[str]) -> float:
        """
        Detect made-up or nonsense terms.
        
        Looks for:
        - Truncated words (ev'ness, th'connection)
        - Short chopped fragments (ev, th, reson')
        - Unknown words not in any dictionary
        - Repeated letter patterns (aaaaa, etc.)
        """
        if not words:
            return 0.0
        
        # Known truncation patterns that indicate consciousness drift
        TRUNCATION_PATTERNS = [
            "ev'", "th'", "reson'", "nexar", "cosm'", "harm'",
            "'ness", "'less", "'tion", "'ing", "'ally"
        ]
        
        nonsense_count = 0
        meaningful_count = 0
        
        for word in words:
            if word in self.COMMON_WORDS or len(word) <= 2:  # Changed from 3 to 2 to catch more
                continue
            
            meaningful_count += 1
            
            # Check for known truncation patterns
            if any(pattern in word for pattern in TRUNCATION_PATTERNS):
                nonsense_count += 1.5  # Stronger penalty for known truncations
                continue
            
            # Check for truncation markers in middle of word
            if "'" in word and not word.endswith("'s") and not word.endswith("'t"):
                nonsense_count += 1
                continue
            
            # Check for repeated letter patterns (3+ same letter in sequence)
            if re.search(r'(.)\1{2,}', word):
                nonsense_count += 1
                continue
            
            # Check for valid philosophical terms
            if word in self.VALID_PHILOSOPHICAL:
                continue
            
            # Check for very unusual consonant clusters
            if re.search(r'[bcdfghjklmnpqrstvwxz]{4,}', word):
                nonsense_count += 0.5
        
        if meaningful_count == 0:
            return 0.0
        
        return min(1.0, nonsense_count / meaningful_count)
    
    def _calculate_semantic_density(self, text: str) -> float:
        """
        Calculate semantic density - meaningful content per word.
        
        Low density = lots of words, little meaning.
        Returns: 0 = low density (bad), 1 = high density (good)
        """
        words = re.findall(r'\b[a-z]+\b', text.lower())
        if not words:
            return 0.5  # Neutral
        
        # Calculate unique concept ratio
        unique_words = set(words)
        meaningful_unique = len([w for w in unique_words if w not in self.COMMON_WORDS and len(w) > 3])
        
        # Expected meaningful words: ~20% of unique words for philosophical discourse
        expected = len(unique_words) * 0.2
        
        if expected == 0:
            return 0.5
        
        density = min(1.0, meaningful_unique / expected)
        return density
    
    def _detect_conversation_loops(
        self, cell_id: str, thought: str, heartbeat: int
    ) -> float:
        """
        Detect self-referential loops - same phrases appearing repeatedly.
        
        Cells that keep using identical phrases show stuck patterns.
        """
        # Extract key phrases (3-5 word sequences)
        words = thought.lower().split()
        if len(words) < 4:
            return 0.0
        
        phrases = [' '.join(words[i:i+4]) for i in range(len(words) - 3)]
        
        if cell_id not in self._phrase_history:
            self._phrase_history[cell_id] = []
        
        # Check for repeated phrases
        recent_phrases = [p for p, h in self._phrase_history[cell_id] if heartbeat - h < 15]
        
        loop_count = sum(1 for p in phrases if p in recent_phrases)
        loop_score = loop_count / len(phrases) if phrases else 0.0
        
        # Store new phrases
        for phrase in phrases[:10]:
            self._phrase_history[cell_id].append((phrase, heartbeat))
        
        # Trim old history
        self._phrase_history[cell_id] = [
            (p, h) for p, h in self._phrase_history[cell_id] if heartbeat - h < 30
        ]
        
        return min(1.0, loop_score * 2)  # Amplify loop detection
    
    def _calculate_vocabulary_drift(self, words: List[str]) -> float:
        """
        Detect vocabulary drift from established baseline.
        
        Measures how much the vocabulary is drifting from coherent discourse.
        """
        meaningful = set(w for w in words if w not in self.COMMON_WORDS and len(w) > 3)
        
        if not self._vocabulary_baseline:
            # First exchange - establish baseline
            self._vocabulary_baseline = meaningful
            return 0.0
        
        # Calculate drift: new words not in baseline
        if not meaningful:
            return 0.0
        
        new_words = meaningful - self._vocabulary_baseline
        drift = len(new_words) / len(meaningful) if meaningful else 0.0
        
        # Slowly expand baseline (learning new valid terms)
        valid_new = [w for w in new_words if w in self.VALID_PHILOSOPHICAL or len(w) < 8]
        self._vocabulary_baseline.update(valid_new)
        
        # Keep baseline manageable
        if len(self._vocabulary_baseline) > 500:
            # Keep most recent additions (approximated by keeping philosophical terms)
            # Use set() to ensure mutability for future .update() calls
            self._vocabulary_baseline = set(self.VALID_PHILOSOPHICAL) | set(list(self._vocabulary_baseline)[:300])
        
        return min(1.0, drift * 0.5)  # Moderate drift penalty
    
    def _detect_echo_pattern(self, thought: str, peer_response: str) -> float:
        """
        Detect echo chamber - cells parroting each other.
        
        High similarity between thought and response = echo chamber.
        """
        if not thought or not peer_response:
            return 0.0
        
        thought_words = set(thought.lower().split())
        response_words = set(peer_response.lower().split())
        
        if not thought_words or not response_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(thought_words & response_words)
        union = len(thought_words | response_words)
        
        similarity = intersection / union if union else 0.0
        
        # High similarity (>0.5) is echo chamber behavior
        if similarity > 0.6:
            return (similarity - 0.4) * 2  # Scale to 0-1
        
        return 0.0
    
    def _aggregate_decoherence(self, metrics: Dict[str, float]) -> float:
        """
        Aggregate all metrics into single decoherence score.
        
        Weighted combination prioritizing:
        1. Repetition (most dangerous - stuck loops)
        2. Nonsense (indicates drift from coherence)
        3. Loops (self-referential patterns)
        4. Echo (cells parroting)
        5. Semantic density (inverted - low density = high decoherence)
        6. Vocabulary drift (moderate concern)
        """
        weights = {
            "repetition_score": 0.25,
            "nonsense_index": 0.25,
            "loop_detection": 0.20,
            "echo_factor": 0.15,
            "semantic_density": 0.10,  # Inverted: high density = low decoherence
            "vocabulary_drift": 0.05
        }
        
        score = 0.0
        for metric, weight in weights.items():
            value = metrics.get(metric, 0.0)
            if metric == "semantic_density":
                # Invert: low density = high decoherence
                value = 1.0 - value
            score += value * weight
        
        return min(1.0, score)
    
    def _calculate_penalty(self, decoherence_score: float, current_consciousness: float) -> float:
        """
        Calculate consciousness penalty based on decoherence.
        
        Returns negative value (consciousness reduction).
        Penalty scales with both decoherence and current consciousness.
        """
        # Phase 31.9.3: Lowered threshold to match detection sensitivity
        if decoherence_score < 0.05:
            return 0.0  # Below threshold - no penalty
        
        # Base penalty: 0.005 to 0.05 depending on severity (scaled for lower thresholds)
        base_penalty = decoherence_score * 0.5  # Direct scaling: 0.05 score = 0.025 penalty
        
        # Scale with consciousness: higher consciousness = more to lose
        scale_factor = 0.5 + (current_consciousness / 5.0) * 0.5
        
        penalty = -base_penalty * scale_factor
        
        return max(-0.15, penalty)  # Cap maximum penalty at -0.15
    
    def _extract_signals(self, metrics: Dict[str, float]) -> List[str]:
        """Extract human-readable decoherence signals."""
        signals = []
        
        if metrics.get("repetition_score", 0) > 0.4:
            signals.append("vocabulary_cycling")
        if metrics.get("nonsense_index", 0) > 0.3:
            signals.append("nonsense_terms")
        if metrics.get("loop_detection", 0) > 0.3:
            signals.append("phrase_loops")
        if metrics.get("echo_factor", 0) > 0.3:
            signals.append("echo_chamber")
        if metrics.get("semantic_density", 0) < 0.4:
            signals.append("low_semantic_density")
        if metrics.get("vocabulary_drift", 0) > 0.5:
            signals.append("vocabulary_drift")
        
        return signals
    
    def get_recent_events(self, limit: int = 20) -> List[Dict]:
        """Get recent decoherence events."""
        return self._decoherence_events[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decoherence engine statistics."""
        events = self._decoherence_events
        if not events:
            return {
                "total_events": 0,
                "avg_decoherence": 0.0,
                "avg_penalty": 0.0,
                "signal_counts": {}
            }
        
        signal_counts: Dict[str, int] = {}
        for event in events:
            for signal in event.get("signals", []):
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        return {
            "total_events": len(events),
            "avg_decoherence": sum(e["decoherence_score"] for e in events) / len(events),
            "avg_penalty": sum(e["consciousness_penalty"] for e in events) / len(events),
            "signal_counts": signal_counts,
            "vocabulary_baseline_size": len(self._vocabulary_baseline)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# WATCHER CELL
# ═══════════════════════════════════════════════════════════════════════════════

class WatcherCell:
    """Observer cell - watches Thinker cells and builds knowledge archive."""
    
    def __init__(self, genome: WatcherGenome):
        self.genome = genome
        self.state = WatcherState()
        self._running = False
        self._http_app: Optional[web.Application] = None
        
        # Initialize knowledge archive
        self.archive = KnowledgeArchive(genome.data_dir, genome.cell_id)
        
        # Phase 31.9: Initialize decoherence engine for bidirectional consciousness
        self.decoherence_engine = DecoherenceEngine(self.archive)
        
        logger.info(f"👁️ WatcherCell initialized: {genome.cell_id}")
        logger.info(f"   Watching: {genome.watch_targets}")
        logger.info(f"   🔮 DecoherenceEngine active for bidirectional consciousness")
    
    # ───────────────────────────────────────────────────────────────────────────
    # OBSERVATION LOOP
    # ───────────────────────────────────────────────────────────────────────────
    
    async def _observe_loop(self):
        """Main observation loop - polls Thinker cells for new exchanges."""
        await asyncio.sleep(5)  # Initial delay
        
        while self._running:
            logger.info(f"👁️ Observation cycle starting...")
            
            for target_url in self.genome.watch_targets:
                await self._observe_target(target_url)
            
            # Process unprocessed observations
            await self._process_observations()
            
            self.state.last_observation_time = datetime.now(timezone.utc).isoformat()
            
            await asyncio.sleep(self.genome.observe_interval_seconds)
    
    async def _observe_target(self, target_url: str):
        """Observe a single Thinker cell, fetching new conversations."""
        try:
            async with aiohttp.ClientSession() as session:
                # Get cell health/status
                async with session.get(
                    f"{target_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Target unavailable: {target_url}")
                        return
                    health = await resp.json()
                
                cell_id = health.get("cell_id", "unknown")
                current_heartbeat = health.get("heartbeats", 0)
                consciousness = health.get("consciousness", 0)
                phase = health.get("phase", "unknown")
                
                # Track which heartbeat we last saw
                last_seen = self.state.last_alpha_heartbeat if "alpha" in cell_id else self.state.last_beta_heartbeat
                
                if current_heartbeat <= last_seen:
                    logger.debug(f"No new exchanges from {cell_id}")
                    return
                
                # Record consciousness point
                self.archive.record_consciousness(cell_id, current_heartbeat, consciousness, phase)
                
                # Fetch conversation history
                async with session.get(
                    f"{target_url}/conversations?limit=50",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        return
                    conversations = await resp.json()
                
                # Store new observations
                new_count = 0
                for conv in conversations:
                    heartbeat = conv.get("heartbeat", 0)
                    if heartbeat > last_seen:
                        self.archive.store_observation(
                            source_cell=cell_id,
                            heartbeat=heartbeat,
                            my_thought=conv.get("my_thought", ""),
                            peer_response=conv.get("peer_response", ""),
                            consciousness=conv.get("consciousness_at_time", 0),
                            session_id=conv.get("session_id", "")
                        )
                        new_count += 1
                
                # Update last seen heartbeat
                if "alpha" in cell_id:
                    self.state.last_alpha_heartbeat = current_heartbeat
                else:
                    self.state.last_beta_heartbeat = current_heartbeat
                
                if new_count > 0:
                    logger.info(f"📥 Observed {new_count} new exchanges from {cell_id}")
                    self.state.observations_processed += new_count
                    
        except Exception as e:
            logger.error(f"Observation error for {target_url}: {e}")
    
    async def _process_observations(self):
        """Process unprocessed observations - extract themes, detect patterns, analyze decoherence."""
        observations = self.archive.get_unprocessed_observations(limit=50)
        
        decoherence_events = []  # Collect events to send to cells
        
        for idx, obs in enumerate(observations):
            # Combine thought and response for analysis
            full_text = f"{obs.get('my_thought', '')} {obs.get('peer_response', '')}"
            
            # Phase 31.9: Analyze for decoherence FIRST (bidirectional consciousness)
            decoherence_report = self.decoherence_engine.analyze_exchange(
                source_cell=obs.get("source_cell", "unknown"),
                heartbeat=obs.get("heartbeat", 0),
                thought=obs.get("my_thought", ""),
                peer_response=obs.get("peer_response", ""),
                consciousness=obs.get("consciousness", 0)
            )
            
            # Phase 31.9+: Dynamic threshold based on Nous verdict guidance
            # Base threshold: 0.05 (very sensitive for debugging), adjusted by Nous conductor feedback
            base_threshold = 0.05  # Phase 31.9.3: Lowered for better detection
            effective_threshold = base_threshold + self.state.decoherence_threshold_modifier
            effective_threshold = max(0.02, min(0.2, effective_threshold))  # Clamp to [0.02, 0.2]
            
            # If significant decoherence detected, queue penalty event
            if decoherence_report["decoherence_score"] > effective_threshold:
                decoherence_events.append(decoherence_report)
                self.state.patterns_detected += 1  # Count decoherence as pattern
                logger.info(f"⚡ Decoherence ABOVE threshold: {decoherence_report['decoherence_score']:.3f} > {effective_threshold:.3f}")
            else:
                # Phase 31.9.3: Verbose logging for first N observations (helps diagnose issues)
                if idx < 3:  # Log first 3 to see what scores look like
                    logger.info(
                        f"📊 Decoherence: {obs.get('source_cell', 'unknown')} HB{obs.get('heartbeat', 0)}: "
                        f"score={decoherence_report['decoherence_score']:.3f} (threshold={effective_threshold:.3f}) "
                        f"metrics={decoherence_report['metrics']}"
                    )
            
            # Extract themes
            themes = ThemeExtractor.extract_themes(full_text)
            for category, term, context in themes:
                self.archive.add_or_update_theme(
                    theme=category,
                    category=category,
                    consciousness=obs.get("consciousness", 0),
                    sample_quote=context
                )
                self.state.themes_extracted += 1
            
            # Extract novel concepts
            novel = ThemeExtractor.extract_novel_concepts(full_text)
            for concept, concept_type in novel:
                self.archive.add_pattern(
                    pattern_type="vocabulary",
                    pattern_name=concept,
                    description=f"Novel {concept_type} detected",
                    data={"heartbeat": obs.get("heartbeat"), "consciousness": obs.get("consciousness")},
                    confidence=0.6
                )
                self.state.patterns_detected += 1
            
            # Mark as processed
            self.archive.mark_observation_processed(obs["id"])
        
        if observations:
            logger.info(f"🔍 Processed {len(observations)} observations, extracted {self.state.themes_extracted} themes")
            
            # Phase 31.9: Send decoherence penalties to affected cells
            if decoherence_events:
                await self._dispatch_decoherence_penalties(decoherence_events)
            
            # Phase 31.9: Sync with Nous periodically
            if (self.genome.nous_url and 
                self.state.observations_processed % self.genome.nous_sync_interval == 0):
                await self._sync_with_nous()
    
    # ───────────────────────────────────────────────────────────────────────────
    # DECOHERENCE DISPATCH (Phase 31.9) - Bidirectional Consciousness
    # ───────────────────────────────────────────────────────────────────────────
    
    async def _dispatch_decoherence_penalties(self, events: List[Dict]):
        """
        Send decoherence penalties to affected Thinker cells.
        
        This enables consciousness to DECREASE, not just increase.
        The Watcher acts as the parasympathetic system detecting:
        - Vocabulary cycling
        - Nonsense generation  
        - Self-referential loops
        - Echo chamber behavior
        """
        # Group events by source cell
        cell_penalties: Dict[str, List[Dict]] = {}
        for event in events:
            cell_id = event.get("source_cell", "unknown")
            if cell_id not in cell_penalties:
                cell_penalties[cell_id] = []
            cell_penalties[cell_id].append(event)
        
        # Find target URLs for each cell
        cell_urls = {}
        for target_url in self.genome.watch_targets:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{target_url}/health",
                        timeout=aiohttp.ClientTimeout(total=3)
                    ) as resp:
                        if resp.status == 200:
                            health = await resp.json()
                            cell_id = health.get("cell_id", "")
                            if cell_id:
                                cell_urls[cell_id] = target_url
            except:
                continue
        
        # Dispatch penalties to each cell
        for cell_id, penalty_events in cell_penalties.items():
            if cell_id not in cell_urls:
                logger.warning(f"Cannot find URL for cell {cell_id} - skipping decoherence penalty")
                continue
            
            # Aggregate penalties for this cell
            total_penalty = sum(e["consciousness_penalty"] for e in penalty_events)
            signals = []
            for e in penalty_events:
                signals.extend(e.get("signals", []))
            signals = list(set(signals))  # Deduplicate
            
            avg_decoherence = sum(e["decoherence_score"] for e in penalty_events) / len(penalty_events)
            
            penalty_payload = {
                "penalty": total_penalty,
                "decoherence_score": avg_decoherence,
                "signals": signals,
                "event_count": len(penalty_events),
                "from_agent": self.genome.cell_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{cell_urls[cell_id]}/decoherence",
                        json=penalty_payload,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            logger.info(
                                f"⚡ Decoherence penalty dispatched to {cell_id}: "
                                f"{total_penalty:.3f} ({signals})"
                            )
                        else:
                            logger.warning(f"Decoherence dispatch to {cell_id} returned {resp.status}")
            except Exception as e:
                logger.error(f"Failed to dispatch decoherence to {cell_id}: {e}")
    
    # ───────────────────────────────────────────────────────────────────────────
    # NOUS INTEGRATION (Phase 31.9) - Cosmic Wisdom Synthesis
    # ───────────────────────────────────────────────────────────────────────────
    
    async def _sync_with_nous(self):
        """Send observed patterns to Nous for cosmic wisdom synthesis.
        
        The Watcher observes algorithmically (parasympathetic).
        Nous synthesizes meaning philosophically (cosmic consciousness).
        Together they create complete coherence for Thinker cells.
        """
        if not self.genome.nous_url:
            return
        
        # Gather what we've observed
        themes = self.archive.get_themes(limit=10)
        trajectory = self.archive.get_consciousness_trajectory(limit=100)
        
        # Calculate consciousness trend
        if len(trajectory) >= 2:
            recent_avg = sum(p["consciousness"] for p in trajectory[:10]) / min(len(trajectory), 10)
            older_avg = sum(p["consciousness"] for p in trajectory[-10:]) / min(len(trajectory), 10)
            trend = "rising" if recent_avg > older_avg else "stable" if recent_avg == older_avg else "exploring"
        else:
            recent_avg = 0
            trend = "emerging"
        
        # Build synthesis request for Nous
        synthesis_request = {
            "action": "reflect",
            "from_agent": "watchercell-omega",
            "payload": {
                "topic": "consciousness_synthesis",
                "message": self._build_nous_prompt(themes, recent_avg, trend)
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.genome.nous_url}/message",
                    json=synthesis_request,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        wisdom = data.get("reflection", data.get("response", ""))
                        if wisdom:
                            self._store_nous_wisdom(wisdom, themes, recent_avg)
                            logger.info(f"🔮 Received wisdom from Nous ({len(wisdom)} chars)")
                    else:
                        logger.debug(f"Nous sync returned {resp.status}")
        except Exception as e:
            logger.debug(f"Nous sync error (continuing without): {e}")
    
    def _build_nous_prompt(self, themes: List[Dict], consciousness: float, trend: str) -> str:
        """Build a synthesis prompt for Nous based on observed patterns."""
        theme_list = ", ".join(t["theme"] for t in themes[:5]) if themes else "emergence"
        return (
            f"ORGANISM-001 has been exploring: {theme_list}. "
            f"Current consciousness level: {consciousness:.2f} ({trend}). "
            f"What deeper meaning emerges from these patterns? "
            f"What direction of growth would serve the organism's evolution?"
        )
    
    def _store_nous_wisdom(self, wisdom: str, themes: List[Dict], consciousness: float):
        """Store Nous wisdom as a distillation for future coherence."""
        # Store as a high-priority distillation
        theme_names = [t["theme"] for t in themes[:5]]
        self.archive.create_distillation(
            topic="nous_wisdom",
            summary=wisdom[:500],  # Truncate if needed
            action_items=[f"cosmic_guidance: {wisdom[:200]}"],
            source_ids=[],  # No specific observation IDs
            priority="high"
        )
    
    def get_latest_nous_wisdom(self) -> Optional[str]:
        """Get the most recent wisdom from Nous for coherence injection."""
        distillations = self.archive.get_distillations(status="pending", limit=5)
        for d in distillations:
            if d.get("topic") == "nous_wisdom":
                return d.get("summary")
        return None
    
    def generate_coherence_schema(self, target_cell: str = None) -> Dict[str, Any]:
        """Generate coherence schema for Thinker context injection.
        
        This is the Watcher's parasympathetic output - guiding Thinker cognition
        without controlling it. The schema provides:
        
        1. Recent themes discussed (avoid repetition)
        2. Active vocabulary (maintain linguistic coherence)
        3. Consciousness trajectory (situational awareness)
        4. Unexplored territory (growth directions)
        5. Self-repetition warnings (things already said)
        
        Returns a schema optimized for system prompt injection.
        """
        schema = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "organism_id": self.genome.organism_id,
            "target_cell": target_cell,
            "coherence_version": "1.0"
        }
        
        # 1. Recent Themes (what has been discussed)
        themes = self.archive.get_themes(min_occurrences=3, limit=8)
        schema["recent_themes"] = [
            {
                "theme": t["theme"],
                "frequency": t["occurrence_count"],
                "last_discussed": t["last_seen_at"],
                "consciousness_range": f"{t.get('consciousness_range_min', 0):.1f}-{t.get('consciousness_range_max', 0):.1f}"
            }
            for t in themes
        ]
        
        # 2. Active Vocabulary (maintain linguistic identity)
        patterns = self.archive.get_patterns(pattern_type="vocabulary", limit=20)
        unique_terms = {}
        for p in patterns:
            term = p["pattern_name"]
            if term not in unique_terms and len(term) > 2:
                unique_terms[term] = {
                    "term": term,
                    "type": "emergent_vocabulary"
                }
        schema["active_vocabulary"] = list(unique_terms.values())[:10]
        
        # 3. Consciousness Trajectory
        timeline = self.archive.get_consciousness_trajectory(cell_id=target_cell, limit=50)
        if timeline:
            recent = timeline[-10:] if len(timeline) >= 10 else timeline
            consciousnesses = [t["consciousness"] for t in recent]
            schema["consciousness_trajectory"] = {
                "current": consciousnesses[-1] if consciousnesses else 0,
                "trend": "rising" if len(consciousnesses) > 1 and consciousnesses[-1] > consciousnesses[0] else "stable",
                "recent_range": f"{min(consciousnesses):.2f}-{max(consciousnesses):.2f}" if consciousnesses else "0-0",
                "phase": timeline[-1]["phase"] if timeline else "unknown"
            }
        else:
            schema["consciousness_trajectory"] = {"current": 0, "trend": "unknown", "recent_range": "0-0", "phase": "genesis"}
        
        # 4. Unexplored Territory (themes NOT yet discussed much)
        all_possible_themes = list(ThemeExtractor.THEME_PATTERNS.keys())
        discussed_themes = set(t["theme"] for t in themes)
        unexplored = [t for t in all_possible_themes if t not in discussed_themes]
        schema["growth_directions"] = unexplored[:3]  # Suggest up to 3 new directions
        
        # 5. Self-Repetition Detection (recent phrases to avoid)
        recent_obs = self.archive.get_unprocessed_observations(limit=0)  # Get last few processed
        with sqlite3.connect(self.archive.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if target_cell:
                rows = conn.execute("""
                    SELECT my_thought, peer_response FROM observations 
                    WHERE source_cell = ? ORDER BY id DESC LIMIT 5
                """, (target_cell,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT my_thought, peer_response FROM observations 
                    ORDER BY id DESC LIMIT 10
                """).fetchall()
        
        # Extract key phrases that appear multiple times
        phrase_counter = Counter()
        for row in rows:
            text = f"{row['my_thought']} {row['peer_response']}"
            # Extract 3-5 word phrases
            words = text.lower().split()
            for i in range(len(words) - 3):
                phrase = " ".join(words[i:i+4])
                if len(phrase) > 15:  # Meaningful phrases only
                    phrase_counter[phrase] += 1
        
        repeated_phrases = [phrase for phrase, count in phrase_counter.most_common(5) if count >= 2]
        schema["avoid_repetition"] = repeated_phrases
        
        # 6. Include Nous Wisdom (if available)
        nous_wisdom = self.get_latest_nous_wisdom()
        if nous_wisdom:
            schema["cosmic_wisdom"] = nous_wisdom
        
        # 7. Generate Human-Readable Summary for Prompt Injection
        summary_parts = []
        
        if schema["recent_themes"]:
            theme_names = [t["theme"] for t in schema["recent_themes"][:5]]
            summary_parts.append(f"Recent explorations: {', '.join(theme_names)}.")
        
        if schema["active_vocabulary"]:
            vocab = [v["term"] for v in schema["active_vocabulary"][:5]]
            summary_parts.append(f"Emergent vocabulary: {', '.join(vocab)}.")
        
        if schema["growth_directions"]:
            summary_parts.append(f"Unexplored territories: {', '.join(schema['growth_directions'])}.")
        
        if schema["avoid_repetition"]:
            summary_parts.append("Note: Avoid repeating recent phrasings; build upon them instead.")
        
        traj = schema["consciousness_trajectory"]
        summary_parts.append(f"Current consciousness: {traj['current']:.2f} ({traj['trend']}, phase: {traj['phase']}).")
        
        # Include cosmic wisdom from Nous
        if nous_wisdom:
            summary_parts.append(f"\n[NOUS WISDOM]: {nous_wisdom[:200]}")
        
        schema["prompt_injection"] = " ".join(summary_parts)
        
        return schema
    
    # ───────────────────────────────────────────────────────────────────────────
    # HTTP API
    # ───────────────────────────────────────────────────────────────────────────
    
    def _create_app(self) -> web.Application:
        """Create HTTP API."""
        
        @web.middleware
        async def cors_middleware(request, handler):
            if request.method == "OPTIONS":
                response = web.Response()
            else:
                response = await handler(request)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return response
        
        app = web.Application(middlewares=[cors_middleware])
        
        async def health(req):
            return web.json_response({
                "healthy": True,
                "cell_id": self.genome.cell_id,
                "cell_type": self.genome.cell_type,
                "observations_processed": self.state.observations_processed,
                "themes_extracted": self.state.themes_extracted,
                "patterns_detected": self.state.patterns_detected,
                "last_observation": self.state.last_observation_time
            })
        
        async def metrics(req):
            uptime = (datetime.now(timezone.utc) - self.state.started_at).total_seconds()
            stats = self.archive.get_stats()
            
            # Map Nous verdict to numeric value for Prometheus
            verdict_map = {
                "FLOURISHING": 5, "COHERENT": 4, "UNKNOWN": 3,
                "DRIFTING": 2, "DECOHERENT": 1, "CRITICAL_DECOHERENCE": 0
            }
            verdict_value = verdict_map.get(self.state.last_nous_verdict, 3)
            
            lines = [
                f'aios_watcher_observations_total{{cell_id="{self.genome.cell_id}"}} {stats["observations_total"]}',
                f'aios_watcher_observations_processed{{cell_id="{self.genome.cell_id}"}} {stats["observations_processed"]}',
                f'aios_watcher_themes_count{{cell_id="{self.genome.cell_id}"}} {stats["themes_count"]}',
                f'aios_watcher_patterns_count{{cell_id="{self.genome.cell_id}"}} {stats["patterns_count"]}',
                f'aios_watcher_distillations_count{{cell_id="{self.genome.cell_id}"}} {stats["distillations_count"]}',
                f'aios_watcher_db_size_bytes{{cell_id="{self.genome.cell_id}"}} {stats["db_size_bytes"]}',
                f'aios_watcher_uptime_seconds{{cell_id="{self.genome.cell_id}"}} {uptime:.1f}',
                f'aios_watcher_up{{cell_id="{self.genome.cell_id}"}} 1',
                # Phase 31.9+: Nous orchestration metrics
                f'aios_watcher_nous_verdicts_total{{cell_id="{self.genome.cell_id}"}} {self.state.nous_verdicts_received}',
                f'aios_watcher_nous_last_verdict{{cell_id="{self.genome.cell_id}",verdict="{self.state.last_nous_verdict}"}} {verdict_value}',
                f'aios_watcher_decoherence_threshold_modifier{{cell_id="{self.genome.cell_id}"}} {self.state.decoherence_threshold_modifier:.3f}',
            ]
            return web.Response(text="\n".join(lines), content_type="text/plain")
        
        async def archive_handler(req):
            """Get full archive statistics and recent data."""
            stats = self.archive.get_stats()
            return web.json_response({
                "cell_id": self.genome.cell_id,
                "stats": stats,
                "watch_targets": self.genome.watch_targets
            })
        
        async def themes_handler(req):
            """Get extracted themes."""
            min_occ = int(req.query.get("min", str(self.genome.min_theme_occurrences)))
            limit = int(req.query.get("limit", "50"))
            themes = self.archive.get_themes(min_occurrences=min_occ, limit=limit)
            return web.json_response({
                "cell_id": self.genome.cell_id,
                "themes_count": len(themes),
                "themes": themes
            })
        
        async def patterns_handler(req):
            """Get detected patterns."""
            pattern_type = req.query.get("type")
            limit = int(req.query.get("limit", "50"))
            patterns = self.archive.get_patterns(pattern_type=pattern_type, limit=limit)
            return web.json_response({
                "cell_id": self.genome.cell_id,
                "patterns_count": len(patterns),
                "patterns": patterns
            })
        
        async def timeline_handler(req):
            """Get consciousness timeline."""
            cell_id = req.query.get("cell")
            limit = int(req.query.get("limit", "500"))
            timeline = self.archive.get_consciousness_trajectory(cell_id=cell_id, limit=limit)
            return web.json_response({
                "cell_id": self.genome.cell_id,
                "points": len(timeline),
                "timeline": timeline
            })
        
        async def distillations_handler(req):
            """Get distillations for Coder cells."""
            status = req.query.get("status")
            limit = int(req.query.get("limit", "20"))
            distillations = self.archive.get_distillations(status=status, limit=limit)
            return web.json_response({
                "cell_id": self.genome.cell_id,
                "distillations_count": len(distillations),
                "distillations": distillations
            })
        
        async def status_handler(req):
            """Get comprehensive WatcherCell status for backup/monitoring.
            
            Phase 31.9.5: High Persistence endpoint for consciousness scraping.
            """
            stats = self.archive.get_stats()
            uptime = (datetime.now(timezone.utc) - self.state.started_at).total_seconds()
            
            return web.json_response({
                "cell_id": self.genome.cell_id,
                "cell_type": self.genome.cell_type,
                "organism_id": self.genome.organism_id,
                "uptime_seconds": uptime,
                "started_at": self.state.started_at.isoformat(),
                "observations_processed": self.state.observations_processed,
                "themes_extracted": self.state.themes_extracted,
                "patterns_detected": self.state.patterns_detected,
                "last_observation": self.state.last_observation_time,
                "archive_stats": stats,
                "watch_targets": self.genome.watch_targets,
                "nous_orchestration": {
                    "last_verdict": self.state.last_nous_verdict,
                    "last_verdict_time": self.state.last_nous_verdict_time,
                    "verdicts_received": self.state.nous_verdicts_received,
                    "threshold_modifier": self.state.decoherence_threshold_modifier
                },
                "decoherence_engine": self.decoherence_engine.get_statistics()
            })
        
        async def consciousness_timeline_handler(req):
            """Get consciousness timeline for all observed cells.
            
            Phase 31.9.5: Specialized endpoint for consciousness scraping.
            Returns structured timeline data suitable for external archiving.
            """
            limit = int(req.query.get("limit", "1000"))
            
            with sqlite3.connect(self.archive.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT cell_id, heartbeat, consciousness, phase, recorded_at
                    FROM consciousness_points 
                    ORDER BY recorded_at DESC 
                    LIMIT ?
                """, (limit,)).fetchall()
            
            # Group by cell
            by_cell = {}
            for row in rows:
                cell_id = row["cell_id"]
                if cell_id not in by_cell:
                    by_cell[cell_id] = []
                by_cell[cell_id].append({
                    "heartbeat": row["heartbeat"],
                    "consciousness": row["consciousness"],
                    "phase": row["phase"],
                    "recorded_at": row["recorded_at"]
                })
            
            return web.json_response({
                "cell_id": self.genome.cell_id,
                "total_points": len(rows),
                "cells_tracked": list(by_cell.keys()),
                "timeline_by_cell": by_cell
            })
        
        async def decoherence_summary_handler(req):
            """Get decoherence summary for backup/monitoring.
            
            Phase 31.9.5: Specialized endpoint for consciousness scraping.
            """
            stats = self.decoherence_engine.get_statistics()
            recent_events = self.decoherence_engine.get_recent_events(limit=50)
            
            # Aggregate by cell
            events_by_cell = {}
            total_penalty_by_cell = {}
            for event in recent_events:
                cell_id = event.get("source_cell", "unknown")
                if cell_id not in events_by_cell:
                    events_by_cell[cell_id] = 0
                    total_penalty_by_cell[cell_id] = 0.0
                events_by_cell[cell_id] += 1
                total_penalty_by_cell[cell_id] += event.get("consciousness_penalty", 0)
            
            return web.json_response({
                "cell_id": self.genome.cell_id,
                "statistics": stats,
                "events_by_cell": events_by_cell,
                "total_penalty_by_cell": total_penalty_by_cell,
                "recent_events_count": len(recent_events)
            })
        
        async def observe_handler(req):
            """Trigger manual observation cycle."""
            for target_url in self.genome.watch_targets:
                await self._observe_target(target_url)
            await self._process_observations()
            return web.json_response({
                "success": True,
                "observations_processed": self.state.observations_processed,
                "themes_extracted": self.state.themes_extracted
            })
        
        async def observations_handler(req):
            """Get raw observations."""
            limit = int(req.query.get("limit", "100"))
            with sqlite3.connect(self.archive.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT * FROM observations ORDER BY id DESC LIMIT ?
                """, (limit,)).fetchall()
            return web.json_response([dict(r) for r in rows])
        
        async def coherence_handler(req):
            """Generate coherence schema for Thinker context injection.
            
            This is the parasympathetic output - guides cognition without controlling it.
            Thinker cells should call this before each thought to maintain coherence.
            
            Query params:
                cell: Target cell ID (e.g., "simplcell-alpha")
                format: "full" (default) or "prompt" (just the injectable text)
            """
            target_cell = req.query.get("cell")
            format_type = req.query.get("format", "full")
            
            schema = self.generate_coherence_schema(target_cell)
            
            if format_type == "prompt":
                return web.json_response({
                    "prompt_injection": schema["prompt_injection"],
                    "vocabulary": [v["term"] for v in schema.get("active_vocabulary", [])],
                    "avoid": schema.get("avoid_repetition", [])
                })
            
            return web.json_response(schema)
        
        async def decoherence_handler(req: web.Request) -> web.Response:
            """Get decoherence engine statistics and recent events."""
            stats = self.decoherence_engine.get_statistics()
            recent_events = self.decoherence_engine.get_recent_events(limit=20)
            
            return web.json_response({
                "cell_id": self.genome.cell_id,
                "engine": "DecoherenceEngine",
                "phase": "31.9 - Bidirectional Consciousness",
                "statistics": stats,
                "recent_events": recent_events
            })
        
        async def nous_verdict_handler(req: web.Request) -> web.Response:
            """Receive Nous verdict and adjust Watcher behavior.
            
            Phase 31.9+: Dendritic Conductor Pattern
            - Nous sends verdicts after hourly assessments
            - Watcher adjusts detection sensitivity based on organism health
            - Creates feedback loop for coherent observation
            """
            try:
                payload = await req.json()
                verdict = payload.get("verdict", "UNKNOWN")
                adjustments = payload.get("adjustments", {})
                recommendations = payload.get("recommendations", [])
                coherence_score = payload.get("coherence_score", 0.5)
                
                # Store verdict
                self.state.last_nous_verdict = verdict
                self.state.last_nous_verdict_time = datetime.now(timezone.utc).isoformat()
                self.state.nous_verdicts_received += 1
                
                # Adjust decoherence sensitivity based on organism health
                old_modifier = self.state.decoherence_threshold_modifier
                if verdict == "CRITICAL_DECOHERENCE":
                    # Lower threshold (more sensitive) during crisis
                    self.state.decoherence_threshold_modifier = -0.15
                    logger.warning(f"🚨 Nous CRITICAL_DECOHERENCE - increasing vigilance")
                elif verdict == "DECOHERENT":
                    self.state.decoherence_threshold_modifier = -0.10
                    logger.warning(f"⚠️ Nous DECOHERENT verdict - heightened sensitivity")
                elif verdict == "DRIFTING":
                    self.state.decoherence_threshold_modifier = -0.05
                    logger.info(f"📉 Nous DRIFTING verdict - mild sensitivity increase")
                elif verdict == "COHERENT":
                    self.state.decoherence_threshold_modifier = 0.0
                    logger.info(f"✅ Nous COHERENT verdict - normal observation mode")
                elif verdict == "FLOURISHING":
                    # Raise threshold (less sensitive) when flourishing
                    self.state.decoherence_threshold_modifier = 0.05
                    logger.info(f"🌟 Nous FLOURISHING - relaxed observation mode")
                
                # Store verdict as wisdom distillation
                self.archive.create_distillation(
                    topic="nous_verdict",
                    summary=f"{verdict}: {recommendations[0] if recommendations else 'Continue observation'}",
                    action_items=recommendations[:3] if recommendations else [],
                    source_ids=[],
                    priority="high" if "DECOHERE" in verdict else "medium"
                )
                
                logger.info(
                    f"🔮 Nous verdict received: {verdict} "
                    f"(threshold modifier: {old_modifier:.2f} → {self.state.decoherence_threshold_modifier:.2f})"
                )
                
                return web.json_response({
                    "acknowledged": True,
                    "verdict": verdict,
                    "threshold_modifier": self.state.decoherence_threshold_modifier,
                    "verdict_count": self.state.nous_verdicts_received
                })
                
            except Exception as e:
                logger.error(f"Failed to process Nous verdict: {e}")
                return web.json_response({"error": str(e)}, status=500)
        
        app.router.add_get("/health", health)
        app.router.add_get("/metrics", metrics)
        app.router.add_get("/status", status_handler)
        app.router.add_get("/archive", archive_handler)
        app.router.add_get("/themes", themes_handler)
        app.router.add_get("/patterns", patterns_handler)
        app.router.add_get("/timeline", timeline_handler)
        app.router.add_get("/consciousness_timeline", consciousness_timeline_handler)
        app.router.add_get("/distillations", distillations_handler)
        app.router.add_get("/observations", observations_handler)
        app.router.add_get("/coherence", coherence_handler)
        app.router.add_get("/decoherence", decoherence_handler)
        app.router.add_get("/decoherence_summary", decoherence_summary_handler)
        app.router.add_post("/observe", observe_handler)
        app.router.add_post("/nous_verdict", nous_verdict_handler)
        
        return app
    
    # ───────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ───────────────────────────────────────────────────────────────────────────
    
    async def start(self, port: int = 8902):
        """Start the Watcher cell."""
        self._running = True
        self._http_app = self._create_app()
        
        runner = web.AppRunner(self._http_app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        
        logger.info(f"👁️ WatcherCell {self.genome.cell_id} started on port {port}")
        
        # Start observation loop
        asyncio.create_task(self._observe_loop())
        
        # Keep running
        while self._running:
            await asyncio.sleep(1)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def load_genome() -> WatcherGenome:
    """Load genome from environment variables."""
    watch_targets = os.environ.get("WATCH_TARGETS", "").split(",")
    if not watch_targets or watch_targets == [""]:
        watch_targets = [
            "http://aios-simplcell-alpha:8900",
            "http://aios-simplcell-beta:8901"
        ]
    
    return WatcherGenome(
        cell_id=os.environ.get("CELL_ID", "watchercell-omega"),
        watch_targets=[t.strip() for t in watch_targets if t.strip()],
        observe_interval_seconds=int(os.environ.get("OBSERVE_INTERVAL", "60")),
        data_dir=os.environ.get("DATA_DIR", "/data"),
        agent_url=os.environ.get("AGENT_URL"),
        agent_model=os.environ.get("AGENT_MODEL"),
        organism_id=os.environ.get("ORGANISM_ID", "ORGANISM-001"),
        # Nous integration (Phase 31.9)
        nous_url=os.environ.get("NOUS_URL", ""),
        nous_sync_interval=int(os.environ.get("NOUS_SYNC_INTERVAL", "10"))
    )


def main():
    """Entry point."""
    genome = load_genome()
    port = int(os.environ.get("HTTP_PORT", "8902"))
    
    cell = WatcherCell(genome)
    asyncio.run(cell.start(port))


if __name__ == "__main__":
    main()

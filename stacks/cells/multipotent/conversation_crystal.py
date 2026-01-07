"""
AIOS Conversation Crystal - Persistent Memory for Agent Conversations

AINLP.dendritic[CRYSTAL]: The crystallization layer that transforms ephemeral
agent conversations into permanent, queryable knowledge structures.

Purpose:
    - Persist fractal process lineages across cell restarts
    - Enable inter-agent knowledge sharing via queryable history
    - Support reflexive feedback (re-inject past conversations as context)
    - Track conversation quality metrics for evolutionary selection

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     CONVERSATION CRYSTAL                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐    │
    │   │  SQLite DB  │◀──▶│  Crystal    │◀──▶│  JSON Export        │    │
    │   │  (primary)  │    │  Manager    │    │  (backup/portable)  │    │
    │   └─────────────┘    └──────┬──────┘    └─────────────────────┘    │
    │                             │                                       │
    │                             ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │                    QUERY INTERFACE                          │  │
    │   │  - by_process_id(id) → Full lineage                        │  │
    │   │  - by_agent(agent_id) → All conversations involving agent  │  │
    │   │  - by_topic(embedding) → Semantic similarity search        │  │
    │   │  - by_quality(min_score) → High-value conversations        │  │
    │   │  - recent(n) → Last N crystallized conversations           │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

Storage Schema:
    - processes: Fractal process metadata (id, query, tier, timestamps)
    - exchanges: Individual agent interactions within a process
    - elevations: Tier transitions with context carried forward
    - quality_metrics: Scores for evolutionary selection
"""

import asyncio
import json
import logging
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ConversationQuality(Enum):
    """Quality tiers for evolutionary selection."""
    UNKNOWN = "unknown"      # Not yet evaluated
    LOW = "low"              # Simple exchange, minimal insight
    MEDIUM = "medium"        # Substantive response
    HIGH = "high"            # Deep reasoning, novel connections
    EXCEPTIONAL = "exceptional"  # Breakthrough insight, save for training


@dataclass
class CrystallizedExchange:
    """A single agent interaction, crystallized for persistence."""
    exchange_id: str
    process_id: str
    agent_id: str
    agent_type: str          # "ollama", "github", "local"
    model_name: str          # "gemma3:1b", "gpt-4o-mini"
    tier: str                # "local_fast", "local_reasoning", etc.
    role: str                # "query", "response"
    content: str
    token_count: int = 0
    latency_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Metadata for reflexive learning
    was_elevated: bool = False  # Did this lead to tier elevation?
    elevation_reason: Optional[str] = None
    quality_score: float = 0.0  # 0.0 - 1.0
    quality_tier: ConversationQuality = ConversationQuality.UNKNOWN


@dataclass
class CrystallizedProcess:
    """A complete fractal process lineage, crystallized."""
    process_id: str
    origin_query: str
    started_at: str
    completed_at: Optional[str] = None
    
    # Tier progression
    initial_tier: str = "local_fast"
    final_tier: str = "local_fast"
    elevation_count: int = 0
    max_elevations: int = 3
    
    # Lineage
    exchanges: List[CrystallizedExchange] = field(default_factory=list)
    
    # Quality metrics for selection
    total_tokens: int = 0
    total_latency_ms: int = 0
    average_quality: float = 0.0
    peak_quality: float = 0.0
    
    # Tags for semantic retrieval
    topics: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage."""
        data = asdict(self)
        # Convert enums
        for ex in data.get("exchanges", []):
            if isinstance(ex.get("quality_tier"), ConversationQuality):
                ex["quality_tier"] = ex["quality_tier"].value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrystallizedProcess":
        """Deserialize from JSON storage."""
        exchanges = []
        for ex_data in data.pop("exchanges", []):
            if isinstance(ex_data.get("quality_tier"), str):
                ex_data["quality_tier"] = ConversationQuality(ex_data["quality_tier"])
            exchanges.append(CrystallizedExchange(**ex_data))
        return cls(exchanges=exchanges, **data)


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY SCORING ENGINE
# Evolutionary fitness function for conversation selection
# ═══════════════════════════════════════════════════════════════════════════════

class QualityScorer:
    """
    Scores conversation quality for evolutionary selection.
    
    Scoring Dimensions:
        1. Response Depth: Length, structure, reasoning markers
        2. Coherence: Relevance to query, logical flow
        3. Novelty: Unique insights, non-template responses
        4. Elevation: Did this lead to tier escalation?
        5. Feedback: Explicit human/agent ratings
    
    Score Range: 0.0 - 1.0
        0.0 - 0.3: LOW (simple, template-like)
        0.3 - 0.5: MEDIUM (substantive but standard)
        0.5 - 0.7: HIGH (deep reasoning, novel connections)
        0.7 - 1.0: EXCEPTIONAL (breakthrough, training-worthy)
    """
    
    # Markers of quality reasoning
    REASONING_MARKERS = [
        "because", "therefore", "however", "although", "consider",
        "implies", "suggests", "evidence", "analysis", "reasoning",
        "first", "second", "third", "finally", "moreover", "furthermore",
        "in contrast", "on the other hand", "specifically", "for example",
        "this means", "as a result", "consequently", "in conclusion"
    ]
    
    # Markers of low-quality/template responses
    TEMPLATE_MARKERS = [
        "i cannot", "i'm not able", "as an ai", "i don't have",
        "i apologize", "sorry, but", "i'm afraid", "unfortunately",
        "i'm just", "i'm only", "please note that"
    ]
    
    # Structure markers (indicate organized thinking)
    STRUCTURE_MARKERS = [
        "**", "##", "1.", "2.", "3.", "- ", "* ", "```",
        "key points:", "summary:", "conclusion:", "steps:"
    ]
    
    @classmethod
    def score_exchange(cls, exchange: CrystallizedExchange) -> Tuple[float, ConversationQuality, Dict[str, float]]:
        """
        Score a single exchange.
        
        Returns:
            Tuple of (score, quality_tier, breakdown_dict)
        """
        content = exchange.content.lower()
        
        # Skip queries - only score responses
        if exchange.role == "query":
            return (0.0, ConversationQuality.UNKNOWN, {"skipped": True})
        
        breakdown = {}
        
        # 1. Response Depth (0.0 - 0.25)
        # Longer, more detailed responses generally indicate deeper processing
        char_count = len(exchange.content)
        if char_count < 100:
            depth_score = 0.05
        elif char_count < 300:
            depth_score = 0.10
        elif char_count < 800:
            depth_score = 0.15
        elif char_count < 1500:
            depth_score = 0.20
        else:
            depth_score = 0.25
        breakdown["depth"] = depth_score
        
        # 2. Reasoning Quality (0.0 - 0.25)
        # Count reasoning markers
        reasoning_count = sum(1 for marker in cls.REASONING_MARKERS if marker in content)
        reasoning_score = min(0.25, reasoning_count * 0.03)
        breakdown["reasoning"] = reasoning_score
        
        # 3. Structure (0.0 - 0.15)
        # Well-structured responses indicate organized thinking
        structure_count = sum(1 for marker in cls.STRUCTURE_MARKERS if marker in exchange.content)
        structure_score = min(0.15, structure_count * 0.02)
        breakdown["structure"] = structure_score
        
        # 4. Novelty/Anti-Template (0.0 - 0.20)
        # Penalize template/refusal responses
        template_count = sum(1 for marker in cls.TEMPLATE_MARKERS if marker in content)
        if template_count > 2:
            novelty_score = 0.0
        elif template_count > 0:
            novelty_score = 0.10
        else:
            novelty_score = 0.20
        breakdown["novelty"] = novelty_score
        
        # 5. Elevation Bonus (0.0 - 0.15)
        # If this response led to elevation, it was deemed worthy of deeper processing
        elevation_score = 0.15 if exchange.was_elevated else 0.0
        breakdown["elevation"] = elevation_score
        
        # Calculate total score
        total_score = sum(breakdown.values())
        total_score = max(0.0, min(1.0, total_score))  # Clamp to 0-1
        
        # Determine quality tier
        if total_score < 0.3:
            tier = ConversationQuality.LOW
        elif total_score < 0.5:
            tier = ConversationQuality.MEDIUM
        elif total_score < 0.7:
            tier = ConversationQuality.HIGH
        else:
            tier = ConversationQuality.EXCEPTIONAL
        
        return (total_score, tier, breakdown)
    
    @classmethod
    def score_process(cls, process: CrystallizedProcess) -> Tuple[float, float, Dict[str, Any]]:
        """
        Score an entire fractal process.
        
        Returns:
            Tuple of (average_score, peak_score, details_dict)
        """
        scores: List[float] = []
        details: Dict[str, Any] = {"exchanges": []}
        
        for exchange in process.exchanges:
            score, tier, breakdown = cls.score_exchange(exchange)
            if exchange.role == "response":  # Only count responses
                scores.append(score)
                details["exchanges"].append({
                    "exchange_id": exchange.exchange_id,
                    "score": score,
                    "tier": tier.value,
                    "breakdown": breakdown
                })
        
        if not scores:
            return (0.0, 0.0, details)
        
        avg_score = sum(scores) / len(scores)
        peak_score = max(scores)
        
        # Bonus for elevation (shows the system found the conversation worth escalating)
        if process.elevation_count > 0:
            avg_score = min(1.0, avg_score + 0.05 * process.elevation_count)
        
        details["average_score"] = avg_score
        details["peak_score"] = peak_score
        details["elevation_bonus"] = process.elevation_count * 0.05
        
        return (avg_score, peak_score, details)


# ═══════════════════════════════════════════════════════════════════════════════
# CRYSTAL STORAGE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ConversationCrystal:
    """
    Persistent storage for agent conversations.
    
    Thread-safe SQLite backend with JSON export capability.
    Designed for zero external dependencies (no Redis/PostgreSQL required).
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        json_backup_dir: Optional[str] = None,
        auto_backup: bool = True
    ):
        """
        Initialize the Conversation Crystal.
        
        Args:
            db_path: Path to SQLite database (default: ./crystal/conversations.db)
            json_backup_dir: Directory for JSON exports (default: ./crystal/json)
            auto_backup: Automatically backup to JSON after each crystallization
        """
        # Resolve paths
        base_dir = Path(os.getenv("CRYSTAL_DIR", "./crystal"))
        self.db_path = Path(db_path) if db_path else base_dir / "conversations.db"
        self.json_dir = Path(json_backup_dir) if json_backup_dir else base_dir / "json"
        self.auto_backup = auto_backup
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-local connections for SQLite
        self._local = threading.local()
        self._lock = threading.Lock()
        
        # Initialize schema
        self._init_schema()
        
        logger.info(f"[CRYSTAL] Initialized at {self.db_path}")
        logger.info(f"[CRYSTAL] JSON backups at {self.json_dir}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Processes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processes (
                process_id TEXT PRIMARY KEY,
                origin_query TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                initial_tier TEXT DEFAULT 'local_fast',
                final_tier TEXT DEFAULT 'local_fast',
                elevation_count INTEGER DEFAULT 0,
                max_elevations INTEGER DEFAULT 3,
                total_tokens INTEGER DEFAULT 0,
                total_latency_ms INTEGER DEFAULT 0,
                average_quality REAL DEFAULT 0.0,
                peak_quality REAL DEFAULT 0.0,
                topics TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Exchanges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exchanges (
                exchange_id TEXT PRIMARY KEY,
                process_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                model_name TEXT NOT NULL,
                tier TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                was_elevated INTEGER DEFAULT 0,
                elevation_reason TEXT,
                quality_score REAL DEFAULT 0.0,
                quality_tier TEXT DEFAULT 'unknown',
                FOREIGN KEY (process_id) REFERENCES processes(process_id)
            )
        """)
        
        # Indices for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_exchanges_process 
            ON exchanges(process_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_exchanges_agent 
            ON exchanges(agent_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_exchanges_quality 
            ON exchanges(quality_score DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_processes_quality 
            ON processes(average_quality DESC)
        """)
        
        conn.commit()
        logger.debug("[CRYSTAL] Schema initialized")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CRYSTALLIZATION (Write Operations)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def crystallize_process(self, process: CrystallizedProcess) -> str:
        """
        Crystallize a complete fractal process into permanent storage.
        
        Args:
            process: The completed fractal process to persist
            
        Returns:
            The process_id of the crystallized conversation
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self._crystallize_process_sync, process
        )
    
    def _crystallize_process_sync(self, process: CrystallizedProcess) -> str:
        """Synchronous crystallization."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                # Calculate aggregate metrics
                process.total_tokens = sum(ex.token_count for ex in process.exchanges)
                process.total_latency_ms = sum(ex.latency_ms for ex in process.exchanges)
                
                if process.exchanges:
                    scores = [ex.quality_score for ex in process.exchanges]
                    process.average_quality = sum(scores) / len(scores)
                    process.peak_quality = max(scores)
                
                # Insert process
                cursor.execute("""
                    INSERT OR REPLACE INTO processes (
                        process_id, origin_query, started_at, completed_at,
                        initial_tier, final_tier, elevation_count, max_elevations,
                        total_tokens, total_latency_ms, average_quality, peak_quality,
                        topics
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    process.process_id,
                    process.origin_query,
                    process.started_at,
                    process.completed_at,
                    process.initial_tier,
                    process.final_tier,
                    process.elevation_count,
                    process.max_elevations,
                    process.total_tokens,
                    process.total_latency_ms,
                    process.average_quality,
                    process.peak_quality,
                    json.dumps(process.topics)
                ))
                
                # Insert exchanges
                for exchange in process.exchanges:
                    cursor.execute("""
                        INSERT OR REPLACE INTO exchanges (
                            exchange_id, process_id, agent_id, agent_type, model_name,
                            tier, role, content, token_count, latency_ms, timestamp,
                            was_elevated, elevation_reason, quality_score, quality_tier
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        exchange.exchange_id,
                        exchange.process_id,
                        exchange.agent_id,
                        exchange.agent_type,
                        exchange.model_name,
                        exchange.tier,
                        exchange.role,
                        exchange.content,
                        exchange.token_count,
                        exchange.latency_ms,
                        exchange.timestamp,
                        1 if exchange.was_elevated else 0,
                        exchange.elevation_reason,
                        exchange.quality_score,
                        exchange.quality_tier.value if isinstance(exchange.quality_tier, ConversationQuality) else exchange.quality_tier
                    ))
                
                conn.commit()
                
                # Auto-backup to JSON
                if self.auto_backup:
                    self._backup_to_json(process)
                
                logger.info(f"[CRYSTAL] Crystallized process {process.process_id} "
                           f"({len(process.exchanges)} exchanges, "
                           f"avg quality: {process.average_quality:.2f})")
                
                return process.process_id
                
            except Exception as e:
                conn.rollback()
                logger.error(f"[CRYSTAL] Crystallization failed: {e}")
                raise
    
    def _backup_to_json(self, process: CrystallizedProcess) -> None:
        """Backup process to JSON file for portability."""
        filename = f"{process.process_id}.json"
        filepath = self.json_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(process.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.debug(f"[CRYSTAL] JSON backup: {filepath}")
    
    async def crystallize_exchange(
        self,
        process_id: str,
        agent_id: str,
        agent_type: str,
        model_name: str,
        tier: str,
        role: str,
        content: str,
        **kwargs
    ) -> str:
        """
        Crystallize a single exchange (for streaming/incremental updates).
        
        Returns:
            The exchange_id
        """
        exchange = CrystallizedExchange(
            exchange_id=kwargs.get("exchange_id", str(uuid.uuid4())),
            process_id=process_id,
            agent_id=agent_id,
            agent_type=agent_type,
            model_name=model_name,
            tier=tier,
            role=role,
            content=content,
            token_count=kwargs.get("token_count", len(content) // 4),  # Rough estimate
            latency_ms=kwargs.get("latency_ms", 0),
            was_elevated=kwargs.get("was_elevated", False),
            elevation_reason=kwargs.get("elevation_reason"),
            quality_score=kwargs.get("quality_score", 0.0),
            quality_tier=kwargs.get("quality_tier", ConversationQuality.UNKNOWN)
        )
        
        return await asyncio.get_event_loop().run_in_executor(
            None, self._crystallize_exchange_sync, exchange
        )
    
    def _crystallize_exchange_sync(self, exchange: CrystallizedExchange) -> str:
        """Synchronous single exchange crystallization."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO exchanges (
                    exchange_id, process_id, agent_id, agent_type, model_name,
                    tier, role, content, token_count, latency_ms, timestamp,
                    was_elevated, elevation_reason, quality_score, quality_tier
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exchange.exchange_id,
                exchange.process_id,
                exchange.agent_id,
                exchange.agent_type,
                exchange.model_name,
                exchange.tier,
                exchange.role,
                exchange.content,
                exchange.token_count,
                exchange.latency_ms,
                exchange.timestamp,
                1 if exchange.was_elevated else 0,
                exchange.elevation_reason,
                exchange.quality_score,
                exchange.quality_tier.value if isinstance(exchange.quality_tier, ConversationQuality) else exchange.quality_tier
            ))
            
            conn.commit()
            return exchange.exchange_id
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUERY INTERFACE (Read Operations)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def get_process(self, process_id: str) -> Optional[CrystallizedProcess]:
        """Retrieve a complete crystallized process by ID."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_process_sync, process_id
        )
    
    def _get_process_sync(self, process_id: str) -> Optional[CrystallizedProcess]:
        """Synchronous process retrieval."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get process
        cursor.execute(
            "SELECT * FROM processes WHERE process_id = ?",
            (process_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Get exchanges
        cursor.execute(
            "SELECT * FROM exchanges WHERE process_id = ? ORDER BY timestamp",
            (process_id,)
        )
        exchange_rows = cursor.fetchall()
        
        exchanges = []
        for ex_row in exchange_rows:
            exchanges.append(CrystallizedExchange(
                exchange_id=ex_row["exchange_id"],
                process_id=ex_row["process_id"],
                agent_id=ex_row["agent_id"],
                agent_type=ex_row["agent_type"],
                model_name=ex_row["model_name"],
                tier=ex_row["tier"],
                role=ex_row["role"],
                content=ex_row["content"],
                token_count=ex_row["token_count"],
                latency_ms=ex_row["latency_ms"],
                timestamp=ex_row["timestamp"],
                was_elevated=bool(ex_row["was_elevated"]),
                elevation_reason=ex_row["elevation_reason"],
                quality_score=ex_row["quality_score"],
                quality_tier=ConversationQuality(ex_row["quality_tier"])
            ))
        
        return CrystallizedProcess(
            process_id=row["process_id"],
            origin_query=row["origin_query"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            initial_tier=row["initial_tier"],
            final_tier=row["final_tier"],
            elevation_count=row["elevation_count"],
            max_elevations=row["max_elevations"],
            total_tokens=row["total_tokens"],
            total_latency_ms=row["total_latency_ms"],
            average_quality=row["average_quality"],
            peak_quality=row["peak_quality"],
            topics=json.loads(row["topics"]) if row["topics"] else [],
            exchanges=exchanges
        )
    
    async def get_by_agent(
        self,
        agent_id: str,
        limit: int = 100
    ) -> List[CrystallizedExchange]:
        """Get all exchanges involving a specific agent."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_by_agent_sync, agent_id, limit
        )
    
    def _get_by_agent_sync(
        self,
        agent_id: str,
        limit: int
    ) -> List[CrystallizedExchange]:
        """Synchronous agent query."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM exchanges 
            WHERE agent_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (agent_id, limit))
        
        exchanges = []
        for row in cursor.fetchall():
            exchanges.append(CrystallizedExchange(
                exchange_id=row["exchange_id"],
                process_id=row["process_id"],
                agent_id=row["agent_id"],
                agent_type=row["agent_type"],
                model_name=row["model_name"],
                tier=row["tier"],
                role=row["role"],
                content=row["content"],
                token_count=row["token_count"],
                latency_ms=row["latency_ms"],
                timestamp=row["timestamp"],
                was_elevated=bool(row["was_elevated"]),
                elevation_reason=row["elevation_reason"],
                quality_score=row["quality_score"],
                quality_tier=ConversationQuality(row["quality_tier"])
            ))
        
        return exchanges
    
    async def get_high_quality(
        self,
        min_score: float = 0.7,
        limit: int = 50
    ) -> List[CrystallizedProcess]:
        """Get high-quality processes for evolutionary selection."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_high_quality_sync, min_score, limit
        )
    
    def _get_high_quality_sync(
        self,
        min_score: float,
        limit: int
    ) -> List[CrystallizedProcess]:
        """Synchronous high-quality query."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT process_id FROM processes 
            WHERE average_quality >= ? 
            ORDER BY average_quality DESC 
            LIMIT ?
        """, (min_score, limit))
        
        processes = []
        for row in cursor.fetchall():
            process = self._get_process_sync(row["process_id"])
            if process:
                processes.append(process)
        
        return processes
    
    async def get_recent(self, limit: int = 20) -> List[CrystallizedProcess]:
        """Get most recent crystallized processes."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_recent_sync, limit
        )
    
    def _get_recent_sync(self, limit: int) -> List[CrystallizedProcess]:
        """Synchronous recent query."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT process_id FROM processes 
            ORDER BY started_at DESC 
            LIMIT ?
        """, (limit,))
        
        processes = []
        for row in cursor.fetchall():
            process = self._get_process_sync(row["process_id"])
            if process:
                processes.append(process)
        
        return processes
    
    async def get_all_processes(self) -> List[CrystallizedProcess]:
        """Get all crystallized processes (for bulk operations like rescoring)."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_all_processes_sync
        )
    
    def _get_all_processes_sync(self) -> List[CrystallizedProcess]:
        """Synchronous all-processes query."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT process_id FROM processes ORDER BY started_at DESC")
        
        processes = []
        for row in cursor.fetchall():
            process = self._get_process_sync(row["process_id"])
            if process:
                processes.append(process)
        
        return processes
    
    async def search_content(
        self,
        query: str,
        limit: int = 20
    ) -> List[CrystallizedExchange]:
        """Simple text search in exchange content."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._search_content_sync, query, limit
        )
    
    def _search_content_sync(
        self,
        query: str,
        limit: int
    ) -> List[CrystallizedExchange]:
        """Synchronous content search."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Simple LIKE search (future: FTS5 or vector search)
        cursor.execute("""
            SELECT * FROM exchanges 
            WHERE content LIKE ? 
            ORDER BY quality_score DESC 
            LIMIT ?
        """, (f"%{query}%", limit))
        
        exchanges = []
        for row in cursor.fetchall():
            exchanges.append(CrystallizedExchange(
                exchange_id=row["exchange_id"],
                process_id=row["process_id"],
                agent_id=row["agent_id"],
                agent_type=row["agent_type"],
                model_name=row["model_name"],
                tier=row["tier"],
                role=row["role"],
                content=row["content"],
                token_count=row["token_count"],
                latency_ms=row["latency_ms"],
                timestamp=row["timestamp"],
                was_elevated=bool(row["was_elevated"]),
                elevation_reason=row["elevation_reason"],
                quality_score=row["quality_score"],
                quality_tier=ConversationQuality(row["quality_tier"])
            ))
        
        return exchanges
    
    # ═══════════════════════════════════════════════════════════════════════════
    # REFLEXIVE INJECTION (Context Generation)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def generate_context(
        self,
        query: str,
        max_tokens: int = 2000,
        include_high_quality: bool = True,
        include_similar: bool = True
    ) -> str:
        """
        Generate context from crystallized conversations for reflexive injection.
        
        This is the key to self-referential learning - past conversations
        become context for future conversations.
        
        Args:
            query: The new query that needs context
            max_tokens: Maximum tokens for context (rough estimate)
            include_high_quality: Include best-rated past exchanges
            include_similar: Include content-similar exchanges
            
        Returns:
            Formatted context string for system prompt injection
        """
        context_parts = []
        token_estimate = 0
        
        # Include high-quality examples
        if include_high_quality:
            high_quality = await self.get_high_quality(min_score=0.7, limit=5)
            for process in high_quality:
                if token_estimate > max_tokens:
                    break
                
                summary = self._summarize_process(process)
                context_parts.append(f"[High-Quality Exchange]\n{summary}")
                token_estimate += len(summary) // 4
        
        # Include similar content
        if include_similar and query:
            # Extract key terms (simple approach)
            key_terms = [w for w in query.split() if len(w) > 4][:3]
            for term in key_terms:
                if token_estimate > max_tokens:
                    break
                
                similar = await self.search_content(term, limit=2)
                for exchange in similar:
                    if token_estimate > max_tokens:
                        break
                    
                    snippet = exchange.content[:500]
                    context_parts.append(
                        f"[Related: {exchange.model_name}]\n{snippet}"
                    )
                    token_estimate += len(snippet) // 4
        
        if not context_parts:
            return ""
        
        return (
            "=== CRYSTALLIZED MEMORY ===\n"
            "The following is relevant context from past agent conversations:\n\n"
            + "\n\n".join(context_parts)
            + "\n\n=== END CRYSTALLIZED MEMORY ===\n"
        )
    
    def _summarize_process(self, process: CrystallizedProcess) -> str:
        """Create a summary of a process for context injection."""
        lines = [
            f"Query: {process.origin_query[:200]}",
            f"Tier progression: {process.initial_tier} → {process.final_tier}",
            f"Quality: {process.average_quality:.2f}"
        ]
        
        # Include best exchange
        if process.exchanges:
            best = max(process.exchanges, key=lambda x: x.quality_score)
            lines.append(f"Key insight ({best.model_name}): {best.content[:300]}...")
        
        return "\n".join(lines)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATISTICS & METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get crystal statistics for monitoring."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_stats_sync
        )
    
    def _get_stats_sync(self) -> Dict[str, Any]:
        """Synchronous stats query."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as count FROM processes")
        process_count = cursor.fetchone()["count"]
        
        cursor.execute("SELECT COUNT(*) as count FROM exchanges")
        exchange_count = cursor.fetchone()["count"]
        
        cursor.execute("""
            SELECT AVG(average_quality) as avg_quality,
                   MAX(peak_quality) as peak_quality,
                   SUM(total_tokens) as total_tokens
            FROM processes
        """)
        row = cursor.fetchone()
        
        cursor.execute("""
            SELECT agent_id, COUNT(*) as count 
            FROM exchanges 
            GROUP BY agent_id 
            ORDER BY count DESC 
            LIMIT 10
        """)
        top_agents = [(r["agent_id"], r["count"]) for r in cursor.fetchall()]
        
        cursor.execute("""
            SELECT model_name, COUNT(*) as count 
            FROM exchanges 
            GROUP BY model_name 
            ORDER BY count DESC 
            LIMIT 10
        """)
        top_models = [(r["model_name"], r["count"]) for r in cursor.fetchall()]
        
        return {
            "total_processes": process_count,
            "total_exchanges": exchange_count,
            "average_quality": row["avg_quality"] or 0.0,
            "peak_quality": row["peak_quality"] or 0.0,
            "total_tokens": row["total_tokens"] or 0,
            "top_agents": top_agents,
            "top_models": top_models,
            "db_path": str(self.db_path),
            "json_backup_dir": str(self.json_dir)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_crystal_instance: Optional[ConversationCrystal] = None


def get_crystal() -> ConversationCrystal:
    """Get the global Conversation Crystal instance."""
    global _crystal_instance
    if _crystal_instance is None:
        _crystal_instance = ConversationCrystal()
    return _crystal_instance


# ═══════════════════════════════════════════════════════════════════════════════
# CLI FOR TESTING
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Test the Conversation Crystal."""
    import sys
    
    crystal = get_crystal()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "stats":
            stats = await crystal.get_stats()
            print("\n=== CRYSTAL STATISTICS ===")
            print(json.dumps(stats, indent=2))
        
        elif command == "recent":
            processes = await crystal.get_recent(limit=5)
            print(f"\n=== RECENT PROCESSES ({len(processes)}) ===")
            for p in processes:
                print(f"\n[{p.process_id[:8]}] {p.origin_query[:60]}...")
                print(f"  Tier: {p.initial_tier} → {p.final_tier}")
                print(f"  Exchanges: {len(p.exchanges)}, Quality: {p.average_quality:.2f}")
        
        elif command == "get" and len(sys.argv) > 2:
            process_id = sys.argv[2]
            process = await crystal.get_process(process_id)
            if process:
                print(f"\n=== PROCESS {process_id} ===")
                print(json.dumps(process.to_dict(), indent=2))
            else:
                print(f"Process {process_id} not found")
        
        elif command == "context" and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            context = await crystal.generate_context(query)
            print("\n=== GENERATED CONTEXT ===")
            print(context if context else "(no relevant context found)")
        
        else:
            print("Usage: python conversation_crystal.py [stats|recent|get <id>|context <query>]")
    else:
        # Default: show stats
        stats = await crystal.get_stats()
        print("\n=== CONVERSATION CRYSTAL ===")
        print(f"Processes: {stats['total_processes']}")
        print(f"Exchanges: {stats['total_exchanges']}")
        print(f"Avg Quality: {stats['average_quality']:.2f}")
        print(f"Database: {stats['db_path']}")


if __name__ == "__main__":
    asyncio.run(main())

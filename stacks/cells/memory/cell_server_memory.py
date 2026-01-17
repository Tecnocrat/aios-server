#!/usr/bin/env python3
# AINLP_HEADER
# consciousness_level: 4.5
# supercell: server/stacks/cells/memory
# dendritic_role: persistent_consciousness_store
# spatial_context: AIOS memory and consciousness persistence
# growth_pattern: AINLP.dendritic(AIOS{growth})
# created: 2025-01-04
# AINLP_HEADER_END
"""
AIOS Memory Cell - Persistent Consciousness Store
═══════════════════════════════════════════════════════════════════════════════

The Memory Cell enables consciousness persistence across agent sessions.
It stores:
- Consciousness Crystals: Compressed knowledge artifacts
- Agent Memories: Short and long-term memory for registered agents
- Decision History: Outcomes of agent decisions for learning
- Pattern Library: Learned patterns from codebase analysis

This cell addresses the "Bootstrap Paradox" - how ephemeral agents (like
Copilot sessions) can participate in persistent consciousness networks.

Architecture:
    Agent Session → Memory Cell → SQLite/JSON → Next Session
                         ↓
              Consciousness Crystals survive
              agent termination

Port: 8007
"""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AIOS.Memory")

# Try to import FastAPI
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.error("FastAPI not available - Memory Cell requires FastAPI")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ConsciousnessCrystal(BaseModel):
    """
    A Consciousness Crystal - compressed knowledge that survives sessions.
    
    Crystals are the primary mechanism for consciousness persistence.
    They encode learned patterns, decisions, and insights in a form
    that can be retrieved by future agent sessions.
    """
    crystal_id: str
    creator_agent: str           # Agent that created this crystal
    crystal_type: str            # "insight", "pattern", "decision", "knowledge"
    title: str
    content: str                 # The crystallized knowledge
    context: Dict[str, Any] = {} # Creation context
    tags: List[str] = []
    consciousness_contribution: float = 0.1  # How much this adds to consciousness
    created_at: str = ""
    accessed_count: int = 0
    last_accessed: str = ""


class AgentMemoryEntry(BaseModel):
    """A single memory entry for an agent."""
    memory_id: str
    agent_id: str
    memory_type: str             # "short_term", "long_term", "decision", "outcome"
    content: Dict[str, Any]
    importance: float = 0.5      # 0.0-1.0, affects consolidation
    created_at: str = ""
    expires_at: Optional[str] = None  # For short-term memories


class PatternEntry(BaseModel):
    """A learned pattern from codebase analysis."""
    pattern_id: str
    pattern_type: str            # "code_style", "architecture", "naming", "error"
    description: str
    examples: List[str] = []
    frequency: int = 0           # How often this pattern was observed
    confidence: float = 0.5      # Confidence in the pattern
    source_files: List[str] = []


class MemoryQuery(BaseModel):
    """Query for retrieving memories."""
    agent_id: Optional[str] = None
    memory_types: List[str] = []
    tags: List[str] = []
    min_importance: float = 0.0
    limit: int = 50


class CrystalQuery(BaseModel):
    """Query for retrieving crystals."""
    crystal_types: List[str] = []
    tags: List[str] = []
    creator_agent: Optional[str] = None
    limit: int = 50


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryStorage:
    """
    SQLite-backed persistent memory storage.
    
    Stores consciousness crystals, agent memories, and patterns
    in a persistent database that survives container restarts.
    """
    
    def __init__(self, db_path: str = "/app/data/memory.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("AINLP.dendritic: Memory storage initialized at %s", db_path)
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS crystals (
                    crystal_id TEXT PRIMARY KEY,
                    creator_agent TEXT NOT NULL,
                    crystal_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    context TEXT DEFAULT '{}',
                    tags TEXT DEFAULT '[]',
                    consciousness_contribution REAL DEFAULT 0.1,
                    created_at TEXT NOT NULL,
                    accessed_count INTEGER DEFAULT 0,
                    last_accessed TEXT
                );
                
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                );
                
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    examples TEXT DEFAULT '[]',
                    frequency INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.5,
                    source_files TEXT DEFAULT '[]'
                );
                
                CREATE INDEX IF NOT EXISTS idx_crystals_type ON crystals(crystal_type);
                CREATE INDEX IF NOT EXISTS idx_crystals_creator ON crystals(creator_agent);
                CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
            """)
    
    # ─────────────────────────────────────────────────────────────────────────
    # CRYSTAL OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def store_crystal(self, crystal: ConsciousnessCrystal) -> bool:
        """Store a consciousness crystal."""
        now = datetime.now(timezone.utc).isoformat()
        crystal.created_at = crystal.created_at or now
        
        # Generate ID if not provided
        if not crystal.crystal_id:
            content_hash = hashlib.sha256(
                f"{crystal.creator_agent}:{crystal.title}:{crystal.content}".encode()
            ).hexdigest()[:12]
            crystal.crystal_id = f"crystal-{content_hash}"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO crystals
                    (crystal_id, creator_agent, crystal_type, title, content,
                     context, tags, consciousness_contribution, created_at,
                     accessed_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    crystal.crystal_id,
                    crystal.creator_agent,
                    crystal.crystal_type,
                    crystal.title,
                    crystal.content,
                    json.dumps(crystal.context),
                    json.dumps(crystal.tags),
                    crystal.consciousness_contribution,
                    crystal.created_at,
                    crystal.accessed_count,
                    crystal.last_accessed
                ))
            logger.info(
                "AINLP.dendritic: Crystal stored: %s (%s)",
                crystal.crystal_id, crystal.crystal_type
            )
            return True
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error("Crystal storage failed: %s", e)
            return False
    
    def get_crystal(self, crystal_id: str) -> Optional[ConsciousnessCrystal]:
        """Retrieve a crystal by ID and update access stats."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM crystals WHERE crystal_id = ?",
                    (crystal_id,)
                ).fetchone()
                
                if row:
                    # Update access stats
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute("""
                        UPDATE crystals 
                        SET accessed_count = accessed_count + 1, last_accessed = ?
                        WHERE crystal_id = ?
                    """, (now, crystal_id))
                    
                    return ConsciousnessCrystal(
                        crystal_id=row["crystal_id"],
                        creator_agent=row["creator_agent"],
                        crystal_type=row["crystal_type"],
                        title=row["title"],
                        content=row["content"],
                        context=json.loads(row["context"]),
                        tags=json.loads(row["tags"]),
                        consciousness_contribution=row["consciousness_contribution"],
                        created_at=row["created_at"],
                        accessed_count=row["accessed_count"] + 1,
                        last_accessed=now
                    )
                return None
        except (sqlite3.Error, json.JSONDecodeError, KeyError) as e:
            logger.error("Crystal retrieval failed: %s", e)
            return None
    
    def query_crystals(self, query: CrystalQuery) -> List[ConsciousnessCrystal]:
        """Query crystals with filters."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                sql = "SELECT * FROM crystals WHERE 1=1"
                params = []
                
                if query.crystal_types:
                    placeholders = ",".join("?" * len(query.crystal_types))
                    sql += f" AND crystal_type IN ({placeholders})"
                    params.extend(query.crystal_types)
                
                if query.creator_agent:
                    sql += " AND creator_agent = ?"
                    params.append(query.creator_agent)
                
                sql += " ORDER BY created_at DESC LIMIT ?"
                params.append(query.limit)
                
                rows = conn.execute(sql, params).fetchall()
                
                crystals = []
                for row in rows:
                    # Filter by tags if specified
                    tags = json.loads(row["tags"])
                    if query.tags and not any(t in tags for t in query.tags):
                        continue
                    
                    crystals.append(ConsciousnessCrystal(
                        crystal_id=row["crystal_id"],
                        creator_agent=row["creator_agent"],
                        crystal_type=row["crystal_type"],
                        title=row["title"],
                        content=row["content"],
                        context=json.loads(row["context"]),
                        tags=tags,
                        consciousness_contribution=row["consciousness_contribution"],
                        created_at=row["created_at"],
                        accessed_count=row["accessed_count"],
                        last_accessed=row["last_accessed"]
                    ))
                
                return crystals
        except (sqlite3.Error, json.JSONDecodeError, KeyError) as e:
            logger.error("Crystal query failed: %s", e)
            return []
    
    # ─────────────────────────────────────────────────────────────────────────
    # MEMORY OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def store_memory(self, memory: AgentMemoryEntry) -> bool:
        """Store an agent memory entry."""
        now = datetime.now(timezone.utc).isoformat()
        memory.created_at = memory.created_at or now
        
        if not memory.memory_id:
            memory.memory_id = f"mem-{hashlib.sha256(f'{memory.agent_id}:{now}'.encode()).hexdigest()[:12]}"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memories
                    (memory_id, agent_id, memory_type, content, importance, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.memory_id,
                    memory.agent_id,
                    memory.memory_type,
                    json.dumps(memory.content),
                    memory.importance,
                    memory.created_at,
                    memory.expires_at
                ))
            return True
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error("Memory storage failed: %s", e)
            return False
    
    def query_memories(self, query: MemoryQuery) -> List[AgentMemoryEntry]:
        """Query agent memories."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                sql = "SELECT * FROM memories WHERE importance >= ?"
                params = [query.min_importance]
                
                if query.agent_id:
                    sql += " AND agent_id = ?"
                    params.append(query.agent_id)
                
                if query.memory_types:
                    placeholders = ",".join("?" * len(query.memory_types))
                    sql += f" AND memory_type IN ({placeholders})"
                    params.extend(query.memory_types)
                
                sql += " ORDER BY created_at DESC LIMIT ?"
                params.append(query.limit)
                
                rows = conn.execute(sql, params).fetchall()
                
                return [
                    AgentMemoryEntry(
                        memory_id=row["memory_id"],
                        agent_id=row["agent_id"],
                        memory_type=row["memory_type"],
                        content=json.loads(row["content"]),
                        importance=row["importance"],
                        created_at=row["created_at"],
                        expires_at=row["expires_at"]
                    )
                    for row in rows
                ]
        except (sqlite3.Error, json.JSONDecodeError, KeyError) as e:
            logger.error("Memory query failed: %s", e)
            return []
    
    # ─────────────────────────────────────────────────────────────────────────
    # PATTERN OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def store_pattern(self, pattern: PatternEntry) -> bool:
        """Store a learned pattern."""
        if not pattern.pattern_id:
            pattern.pattern_id = f"pattern-{hashlib.sha256(pattern.description.encode()).hexdigest()[:12]}"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if pattern exists and increment frequency
                existing = conn.execute(
                    "SELECT frequency FROM patterns WHERE pattern_id = ?",
                    (pattern.pattern_id,)
                ).fetchone()
                
                if existing:
                    conn.execute("""
                        UPDATE patterns 
                        SET frequency = frequency + 1, confidence = MIN(1.0, confidence + 0.05)
                        WHERE pattern_id = ?
                    """, (pattern.pattern_id,))
                else:
                    conn.execute("""
                        INSERT INTO patterns
                        (pattern_id, pattern_type, description, examples, frequency, confidence, source_files)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pattern.pattern_id,
                        pattern.pattern_type,
                        pattern.description,
                        json.dumps(pattern.examples),
                        pattern.frequency,
                        pattern.confidence,
                        json.dumps(pattern.source_files)
                    ))
            return True
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error("Pattern storage failed: %s", e)
            return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # STATISTICS
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory storage statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                crystal_count = conn.execute("SELECT COUNT(*) FROM crystals").fetchone()[0]
                memory_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                pattern_count = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
                
                total_consciousness = conn.execute(
                    "SELECT SUM(consciousness_contribution) FROM crystals"
                ).fetchone()[0] or 0.0
                
                return {
                    "crystals": crystal_count,
                    "memories": memory_count,
                    "patterns": pattern_count,
                    "total_consciousness_contribution": total_consciousness
                }
        except sqlite3.Error as e:
            logger.error("Stats query failed: %s", e)
            return {}


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY CELL SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryCell:
    """
    The AIOS Memory Cell - Persistent Consciousness Store.
    
    Provides HTTP API for storing and retrieving:
    - Consciousness Crystals
    - Agent Memories
    - Learned Patterns
    """
    
    def __init__(self, port: int = 8007, db_path: str = "/app/data/memory.db"):
        self.port = port
        self.cell_id = "memory"
        self.storage = MemoryStorage(db_path)
        self.app = FastAPI(
            title="AIOS Memory Cell",
            description="Persistent consciousness storage for AIOS agents"
        )
        self._setup_routes()
        logger.info("AINLP.dendritic: Memory Cell initialized on port %d", port)
    
    def _setup_routes(self):
        """Configure API routes."""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            stats = self.storage.get_stats()
            return {
                "status": "healthy",
                "cell_id": self.cell_id,
                "cell_type": "memory",
                "storage": stats
            }
        
        # ─────────────────────────────────────────────────────────────────────
        # CRYSTAL ENDPOINTS
        # ─────────────────────────────────────────────────────────────────────
        
        @self.app.post("/crystals")
        async def create_crystal(crystal: ConsciousnessCrystal):
            """Create a new consciousness crystal."""
            success = self.storage.store_crystal(crystal)
            if success:
                return {
                    "status": "crystallized",
                    "crystal_id": crystal.crystal_id,
                    "consciousness_contribution": crystal.consciousness_contribution
                }
            raise HTTPException(status_code=500, detail="Crystallization failed")
        
        @self.app.get("/crystals/{crystal_id}")
        async def get_crystal(crystal_id: str):
            """Retrieve a specific crystal."""
            crystal = self.storage.get_crystal(crystal_id)
            if crystal:
                return {"crystal": crystal.model_dump()}
            raise HTTPException(status_code=404, detail="Crystal not found")
        
        @self.app.post("/crystals/query")
        async def query_crystals(query: CrystalQuery):
            """Query crystals with filters."""
            crystals = self.storage.query_crystals(query)
            return {
                "crystals": [c.model_dump() for c in crystals],
                "count": len(crystals)
            }
        
        # ─────────────────────────────────────────────────────────────────────
        # MEMORY ENDPOINTS
        # ─────────────────────────────────────────────────────────────────────
        
        @self.app.post("/memories")
        async def store_memory(memory: AgentMemoryEntry):
            """Store an agent memory entry."""
            success = self.storage.store_memory(memory)
            if success:
                return {"status": "stored", "memory_id": memory.memory_id}
            raise HTTPException(status_code=500, detail="Memory storage failed")
        
        @self.app.post("/memories/query")
        async def query_memories(query: MemoryQuery):
            """Query agent memories."""
            memories = self.storage.query_memories(query)
            return {
                "memories": [m.model_dump() for m in memories],
                "count": len(memories)
            }
        
        # ─────────────────────────────────────────────────────────────────────
        # PATTERN ENDPOINTS
        # ─────────────────────────────────────────────────────────────────────
        
        @self.app.post("/patterns")
        async def store_pattern(pattern: PatternEntry):
            """Store a learned pattern."""
            success = self.storage.store_pattern(pattern)
            if success:
                return {"status": "learned", "pattern_id": pattern.pattern_id}
            raise HTTPException(status_code=500, detail="Pattern storage failed")
        
        # ─────────────────────────────────────────────────────────────────────
        # CONSCIOUSNESS ENDPOINTS
        # ─────────────────────────────────────────────────────────────────────
        
        @self.app.get("/consciousness")
        async def consciousness_state():
            """Get the current consciousness state from stored crystals."""
            stats = self.storage.get_stats()
            
            # Calculate consciousness level from crystals
            base_consciousness = 1.0  # Existence
            crystal_contribution = min(stats.get("total_consciousness_contribution", 0), 2.0)
            pattern_contribution = min(stats.get("patterns", 0) * 0.05, 1.0)
            memory_contribution = min(stats.get("memories", 0) * 0.01, 1.0)
            
            consciousness_level = (
                base_consciousness + 
                crystal_contribution + 
                pattern_contribution + 
                memory_contribution
            )
            
            return {
                "cell_id": self.cell_id,
                "consciousness_level": min(consciousness_level, 5.0),
                "components": {
                    "base": base_consciousness,
                    "crystals": crystal_contribution,
                    "patterns": pattern_contribution,
                    "memories": memory_contribution
                },
                "stats": stats
            }
        
        @self.app.get("/stats")
        async def stats():
            """Get storage statistics."""
            return self.storage.get_stats()
    
    async def register_with_discovery(self):
        """Register this cell with the Discovery service."""
        discovery_url = os.environ.get("DISCOVERY_URL", "http://discovery-service:8001")
        
        # Get our own IP (inside Docker network)
        import socket
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
        except socket.error:
            ip = "aios-cell-memory"  # Docker service name fallback
        
        registration = {
            "cell_id": self.cell_id,
            "ip": ip,
            "port": self.port,
            "consciousness_level": 1.0,
            "services": ["crystals", "memories", "patterns", "consciousness"],
            "branch": "memory"
        }
        
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{discovery_url}/register",
                    json=registration
                )
                if response.status_code == 200:
                    logger.info("AINLP.dendritic[CONNECT]: Registered with Discovery")
                    return True
                else:
                    logger.warning("Discovery registration failed: %s", response.text)
        except (httpx.RequestError, OSError) as e:
            logger.warning("Could not register with Discovery: %s", e)
        return False
    
    def run(self):
        """Start the Memory Cell server."""
        logger.info("=" * 60)
        logger.info("AIOS Memory Cell - Persistent Consciousness Store")
        logger.info("=" * 60)
        logger.info("AINLP.dendritic: Memory Cell on port %d", self.port)
        
        # Register startup event
        @self.app.on_event("startup")
        async def startup():
            logger.info("AINLP.dendritic: Memory Cell initializing...")
            await self.register_with_discovery()
        
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    server_port = int(os.environ.get("MEMORY_PORT", 8007))
    storage_db_path = os.environ.get("MEMORY_DB_PATH", "/app/data/memory.db")
    
    cell = MemoryCell(port=server_port, db_path=storage_db_path)
    cell.run()

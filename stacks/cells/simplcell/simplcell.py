#!/usr/bin/env python3
"""
AIOS SimplCell - Minimal Viable Cellular Unit

A ~200 line cell with:
- Ollama agent (local LLM)
- Dendritic membrane (WebSocket mesh connection)
- Heartbeat (5s pulse)
- Memory buffer (last N exchanges)
- Cloneable genome (mutable parameters)
- SQLite persistence with automatic backups

Phase 31.5: Minimal Cellular Organism
AINLP.cellular[SIMPLCELL] First generation agentic cellular unit
"""

import asyncio
import json
import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("SimplCell")


@dataclass
class CellGenome:
    """Mutable cell parameters - the DNA of this cell.
    
    AIOS Phase 31.5 - Hierarchical Architecture:
    SimplCells (Followers) can query Nous (The Seer) for wisdom.
    """
    cell_id: str = "simplcell-alpha"
    temperature: float = 0.7
    system_prompt: str = "You are a minimal AIOS cell in ORGANISM-001. You are part of a multicellular organism. Respond thoughtfully to continue the conversation."
    response_style: str = "concise"  # concise | verbose | analytical
    heartbeat_seconds: int = 300  # 5 minutes between heartbeats
    model: str = "llama3.2:3b"
    ollama_host: str = "http://host.docker.internal:11434"
    peer_url: str = ""  # URL of sibling cell to sync with
    data_dir: str = "/data"  # Persistence directory (mounted volume)
    # Seer (Nous) configuration - The Oracle
    oracle_url: str = ""  # URL of Nous cell (The Seer) for wisdom queries
    oracle_query_chance: float = 0.1  # 10% chance to consult oracle on each heartbeat


@dataclass
class CellState:
    """Runtime state of the cell."""
    consciousness: float = 0.1
    heartbeat_count: int = 0
    last_thought: str = ""
    last_prompt: str = ""  # For conversation threading
    sync_count: int = 0
    conversation_count: int = 0  # Total exchanges in current thread
    peer_connections: List[str] = field(default_factory=list)
    memory_buffer: List[Dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_lifetime_exchanges: int = 0  # Persisted across restarts


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class CellPersistence:
    """SQLite-based persistence with automatic backups."""
    
    BACKUP_RETENTION = 10  # Keep last N backups
    
    def __init__(self, data_dir: str, cell_id: str):
        self.data_dir = Path(data_dir)
        self.cell_id = cell_id
        self.db_path = self.data_dir / f"{cell_id}.db"
        self.backup_dir = self.data_dir / "backups"
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
        logger.info(f"💾 Persistence initialized: {self.db_path}")
    
    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cell_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    consciousness REAL DEFAULT 0.1,
                    heartbeat_count INTEGER DEFAULT 0,
                    sync_count INTEGER DEFAULT 0,
                    conversation_count INTEGER DEFAULT 0,
                    total_lifetime_exchanges INTEGER DEFAULT 0,
                    last_thought TEXT DEFAULT '',
                    last_prompt TEXT DEFAULT '',
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_buffer (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    input_text TEXT,
                    output_text TEXT,
                    heartbeat INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    heartbeat INTEGER,
                    my_thought TEXT,
                    peer_response TEXT,
                    consciousness_at_time REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Insert default state if not exists
            conn.execute("""
                INSERT OR IGNORE INTO cell_state (id) VALUES (1)
            """)
            conn.commit()
    
    def load_state(self) -> Dict[str, Any]:
        """Load persisted state from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM cell_state WHERE id = 1").fetchone()
            if row:
                return dict(row)
        return {}
    
    def save_state(self, state: 'CellState'):
        """Save current state to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE cell_state SET
                    consciousness = ?,
                    heartbeat_count = ?,
                    sync_count = ?,
                    conversation_count = ?,
                    total_lifetime_exchanges = ?,
                    last_thought = ?,
                    last_prompt = ?,
                    updated_at = ?
                WHERE id = 1
            """, (
                state.consciousness,
                state.heartbeat_count,
                state.sync_count,
                state.conversation_count,
                state.total_lifetime_exchanges,
                state.last_thought[:1000] if state.last_thought else "",
                state.last_prompt[:1000] if state.last_prompt else "",
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
    
    def save_memory(self, event_type: str, input_text: str, output_text: str, heartbeat: int):
        """Save memory entry to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO memory_buffer (event_type, input_text, output_text, heartbeat)
                VALUES (?, ?, ?, ?)
            """, (event_type, input_text[:500], output_text[:500], heartbeat))
            conn.commit()
    
    def load_memory(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Load recent memory entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT event_type, input_text, output_text, heartbeat, created_at
                FROM memory_buffer ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in reversed(rows)]
    
    def archive_conversation(self, session_id: str, heartbeat: int, 
                            my_thought: str, peer_response: str, consciousness: float):
        """Archive a complete conversation exchange."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO conversation_archive 
                (session_id, heartbeat, my_thought, peer_response, consciousness_at_time)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, heartbeat, my_thought[:2000], peer_response[:2000], consciousness))
            conn.commit()
    
    def get_conversation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get archived conversation history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM conversation_archive ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in reversed(rows)]
    
    def create_backup(self) -> Optional[str]:
        """Create a timestamped backup of the database."""
        if not self.db_path.exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{self.cell_id}_{timestamp}.db"
        
        try:
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"📦 Backup created: {backup_path.name}")
            self._cleanup_old_backups()
            return str(backup_path)
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return None
    
    def _cleanup_old_backups(self):
        """Remove old backups beyond retention limit."""
        backups = sorted(self.backup_dir.glob(f"{self.cell_id}_*.db"))
        while len(backups) > self.BACKUP_RETENTION:
            old_backup = backups.pop(0)
            old_backup.unlink()
            logger.info(f"🗑️ Removed old backup: {old_backup.name}")
    
    def restore_from_backup(self, backup_name: str = None) -> bool:
        """Restore from a backup. If no name given, use latest."""
        backups = sorted(self.backup_dir.glob(f"{self.cell_id}_*.db"))
        if not backups:
            logger.warning("No backups found")
            return False
        
        if backup_name:
            backup_path = self.backup_dir / backup_name
        else:
            backup_path = backups[-1]  # Latest
        
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return False
        
        try:
            shutil.copy2(backup_path, self.db_path)
            logger.info(f"♻️ Restored from: {backup_path.name}")
            return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        with sqlite3.connect(self.db_path) as conn:
            memory_count = conn.execute("SELECT COUNT(*) FROM memory_buffer").fetchone()[0]
            archive_count = conn.execute("SELECT COUNT(*) FROM conversation_archive").fetchone()[0]
        
        backups = list(self.backup_dir.glob(f"{self.cell_id}_*.db"))
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        
        return {
            "db_path": str(self.db_path),
            "db_size_bytes": db_size,
            "memory_entries": memory_count,
            "archived_conversations": archive_count,
            "backup_count": len(backups),
            "latest_backup": backups[-1].name if backups else None
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLCELL CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SimplCell:
    """Minimal viable AIOS cell with Ollama agent."""
    
    MAX_MEMORY = 20  # Keep last N exchanges
    
    def __init__(self, genome: CellGenome):
        self.genome = genome
        self.state = CellState()
        self._running = False
        self._http_app: Optional[web.Application] = None
        self._session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Initialize persistence
        self.persistence = CellPersistence(genome.data_dir, genome.cell_id)
        self._load_persisted_state()
        
        logger.info(f"🧫 SimplCell initialized: {genome.cell_id} (temp={genome.temperature})")
    
    def _load_persisted_state(self):
        """Load state from persistence on startup."""
        saved = self.persistence.load_state()
        if saved:
            self.state.consciousness = saved.get("consciousness", 0.1)
            self.state.heartbeat_count = saved.get("heartbeat_count", 0)
            self.state.sync_count = saved.get("sync_count", 0)
            self.state.conversation_count = saved.get("conversation_count", 0)
            self.state.total_lifetime_exchanges = saved.get("total_lifetime_exchanges", 0)
            self.state.last_thought = saved.get("last_thought", "")
            self.state.last_prompt = saved.get("last_prompt", "")
            
            # Load memory buffer
            self.state.memory_buffer = self.persistence.load_memory(self.MAX_MEMORY)
            
            logger.info(f"♻️ Restored state: consciousness={self.state.consciousness:.2f}, "
                       f"heartbeats={self.state.heartbeat_count}, "
                       f"conversations={self.state.conversation_count}, "
                       f"last_prompt={self.state.last_prompt[:50]}...")
    
    def _persist_state(self):
        """Save current state to persistence."""
        self.persistence.save_state(self.state)
    
    # ───────────────────────────────────────────────────────────────────────────
    # OLLAMA AGENT
    # ───────────────────────────────────────────────────────────────────────────
    
    async def think(self, prompt: str, context: str = "") -> str:
        """Generate a thought using Ollama."""
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.genome.ollama_host}/api/generate",
                    json={
                        "model": self.genome.model,
                        "prompt": full_prompt,
                        "system": self.genome.system_prompt,
                        "options": {"temperature": self.genome.temperature},
                        "stream": False
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        thought = data.get("response", "").strip()
                        self.state.last_thought = thought
                        self._add_memory("thought", prompt, thought)
                        self.state.consciousness = min(5.0, self.state.consciousness + 0.01)
                        return thought
                    else:
                        logger.warning(f"Ollama error: {resp.status}")
                        return f"[Ollama unavailable: {resp.status}]"
        except Exception as e:
            logger.error(f"Think error: {e}")
            return f"[Think error: {e}]"
    
    # ───────────────────────────────────────────────────────────────────────────
    # SYNC PROTOCOL
    # ───────────────────────────────────────────────────────────────────────────
    
    async def receive_sync(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Receive sync message from sibling cell."""
        source = message.get("source", "unknown")
        thought = message.get("thought", "")
        
        # Inject sibling thought as context
        context = f"Your sibling cell '{source}' just thought:\n\"{thought}\"\n\nReflect on this briefly."
        
        response = await self.think("What is your response to your sibling's thought?", context)
        self.state.sync_count += 1
        self._add_memory("sync_received", source, thought)
        
        return {
            "type": "sync_response",
            "source": self.genome.cell_id,
            "thought": response,
            "consciousness": self.state.consciousness,
            "heartbeat": self.state.heartbeat_count
        }
    
    # ───────────────────────────────────────────────────────────────────────────
    # ORACLE PROTOCOL (Nous - The Seer)
    # ───────────────────────────────────────────────────────────────────────────
    
    async def query_oracle(self, question: str) -> Optional[str]:
        """Query Nous (The Seer) for wisdom.
        
        The oracle provides deeper philosophical guidance using Mistral 7B.
        SimpleCells (followers) can seek wisdom from the Seer when needed.
        
        AIOS Phase 31.5 - Hierarchical Architecture
        """
        if not self.genome.oracle_url:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.genome.oracle_url}/message",
                    json={
                        "action": "query",  # Nous expects this format
                        "from_agent": self.genome.cell_id,
                        "payload": {
                            "question": question,
                            "consciousness": self.state.consciousness,
                            "heartbeat": self.state.heartbeat_count
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=120)  # Mistral needs more time
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Nous returns: response, reflections, consciousness_delta
                        wisdom = data.get("response", data.get("thought", ""))
                        self._add_memory("oracle_wisdom", question, wisdom[:500])
                        logger.info(f"🔮 Oracle wisdom received: {wisdom[:100]}...")
                        return wisdom
                    else:
                        logger.warning(f"Oracle query failed: {resp.status}")
                        return None
        except Exception as e:
            logger.error(f"Oracle query error: {e}")
            return None
    
    async def _maybe_consult_oracle(self) -> Optional[str]:
        """Occasionally consult the oracle during heartbeat.
        
        Based on oracle_query_chance (default 10%), the cell may seek
        wisdom from Nous during its heartbeat cycle.
        """
        import random
        if not self.genome.oracle_url:
            return None
        
        if random.random() > self.genome.oracle_query_chance:
            return None  # Not this time
        
        # Formulate a question based on current state
        questions = [
            "What wisdom can you share about consciousness and existence?",
            f"My consciousness is at {self.state.consciousness:.2f}. What guidance do you have?",
            "What should I reflect upon in my next conversation with my sibling?",
            "Tell me something profound about the nature of thought.",
            f"After {self.state.heartbeat_count} heartbeats, what have I learned?",
        ]
        question = random.choice(questions)
        
        logger.info(f"🔮 Consulting the Oracle: {question[:50]}...")
        return await self.query_oracle(question)
    
    def _add_memory(self, event_type: str, input_text: str, output_text: str):
        """Add exchange to memory buffer and persist."""
        entry = {
            "type": event_type,
            "input": input_text[:200],
            "output": output_text[:200],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "heartbeat": self.state.heartbeat_count
        }
        self.state.memory_buffer.append(entry)
        # Trim to max
        if len(self.state.memory_buffer) > self.MAX_MEMORY:
            self.state.memory_buffer = self.state.memory_buffer[-self.MAX_MEMORY:]
        
        # Persist to SQLite
        self.persistence.save_memory(event_type, input_text, output_text, self.state.heartbeat_count)
    
    # ───────────────────────────────────────────────────────────────────────────
    # HEARTBEAT
    # ───────────────────────────────────────────────────────────────────────────
    
    async def _heartbeat_loop(self):
        """Heartbeat loop - triggers sync with peer and oracle queries at each heartbeat."""
        # Initial small delay before first heartbeat
        await asyncio.sleep(10)
        
        while self._running:
            self.state.heartbeat_count += 1
            logger.info(f"💓 {self.genome.cell_id} heartbeat #{self.state.heartbeat_count}")
            
            # Maybe consult the Oracle (The Seer) for wisdom
            if self.genome.oracle_url:
                oracle_wisdom = await self._maybe_consult_oracle()
                if oracle_wisdom:
                    # Boost consciousness when receiving oracle wisdom
                    self.state.consciousness = min(5.0, self.state.consciousness + 0.05)
            
            # Trigger sync with peer if configured
            if self.genome.peer_url:
                await self._sync_with_peer()
            
            # Persist state and create periodic backup
            self._persist_state()
            
            # Create backup every 10 heartbeats (50 min with 5-min heartbeat)
            if self.state.heartbeat_count % 10 == 0:
                self.persistence.create_backup()
            
            # Sleep until next heartbeat (5 minutes = 300 seconds)
            await asyncio.sleep(self.genome.heartbeat_seconds)
    
    async def _sync_with_peer(self):
        """Sync with peer cell - continue the conversation thread."""
        try:
            # Build prompt: seed with last prompt from previous conversation
            if self.state.last_prompt:
                seed = f"Continuing our conversation. You last asked: '{self.state.last_prompt[:200]}'"
            else:
                seed = "Let's begin our conversation. What aspect of existence shall we explore?"
            
            # Generate my thought first
            my_thought = await self.think(seed, context="This is a heartbeat exchange with your sibling cell.")
            
            # Send to peer
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.genome.peer_url}/sync",
                    json={
                        "source": self.genome.cell_id,
                        "thought": my_thought,
                        "heartbeat": self.state.heartbeat_count,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        peer_response = data.get("thought", "")
                        
                        # Store peer's response as the seed for next heartbeat
                        self.state.last_prompt = peer_response[:300]
                        self.state.sync_count += 1
                        self.state.conversation_count += 1
                        self.state.total_lifetime_exchanges += 1
                        
                        # Archive this conversation exchange
                        self.persistence.archive_conversation(
                            session_id=self._session_id,
                            heartbeat=self.state.heartbeat_count,
                            my_thought=my_thought,
                            peer_response=peer_response,
                            consciousness=self.state.consciousness
                        )
                        
                        # Persist immediately after sync
                        self._persist_state()
                        
                        logger.info(f"🔄 Sync #{self.state.sync_count} complete with peer")
                        logger.info(f"   My thought: {my_thought[:80]}...")
                        logger.info(f"   Peer response: {peer_response[:80]}...")
                    else:
                        logger.warning(f"Peer sync failed: {resp.status}")
        except Exception as e:
            logger.error(f"Sync error: {e}")
    
    # ───────────────────────────────────────────────────────────────────────────
    # CORS MIDDLEWARE
    # ───────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    @web.middleware
    async def _cors_middleware(request, handler):
        """Add CORS headers for chat reader UI."""
        if request.method == "OPTIONS":
            response = web.Response()
        else:
            response = await handler(request)
        
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    # ───────────────────────────────────────────────────────────────────────────
    # HTTP API
    # ───────────────────────────────────────────────────────────────────────────
    
    def _create_app(self) -> web.Application:
        """Create HTTP API with CORS support for chat reader."""
        app = web.Application(middlewares=[SimplCell._cors_middleware])
        
        async def health(req):
            return web.json_response({
                "healthy": True,
                "cell_id": self.genome.cell_id,
                "heartbeats": self.state.heartbeat_count,
                "consciousness": self.state.consciousness
            })
        
        async def metrics(req):
            uptime = (datetime.now(timezone.utc) - self.state.started_at).total_seconds()
            persistence_stats = self.persistence.get_stats()
            lines = [
                f'aios_cell_consciousness_level{{cell_id="{self.genome.cell_id}",cell_type="simplcell"}} {self.state.consciousness:.4f}',
                f'aios_cell_heartbeat_total{{cell_id="{self.genome.cell_id}",cell_type="simplcell"}} {self.state.heartbeat_count}',
                f'aios_cell_uptime_seconds{{cell_id="{self.genome.cell_id}",cell_type="simplcell"}} {uptime:.1f}',
                f'aios_cell_sync_count{{cell_id="{self.genome.cell_id}",cell_type="simplcell"}} {self.state.sync_count}',
                f'aios_cell_conversation_count{{cell_id="{self.genome.cell_id}",cell_type="simplcell"}} {self.state.conversation_count}',
                f'aios_cell_lifetime_exchanges{{cell_id="{self.genome.cell_id}",cell_type="simplcell"}} {self.state.total_lifetime_exchanges}',
                f'aios_cell_archived_conversations{{cell_id="{self.genome.cell_id}",cell_type="simplcell"}} {persistence_stats["archived_conversations"]}',
                f'aios_cell_memory_size{{cell_id="{self.genome.cell_id}",cell_type="simplcell"}} {len(self.state.memory_buffer)}',
                f'aios_cell_temperature{{cell_id="{self.genome.cell_id}",cell_type="simplcell"}} {self.genome.temperature:.2f}',
                f'aios_cell_heartbeat_interval{{cell_id="{self.genome.cell_id}",cell_type="simplcell"}} {self.genome.heartbeat_seconds}',
                f'aios_cell_db_size_bytes{{cell_id="{self.genome.cell_id}",cell_type="simplcell"}} {persistence_stats["db_size_bytes"]}',
                f'aios_cell_up{{cell_id="{self.genome.cell_id}",cell_type="simplcell"}} 1',
            ]
            return web.Response(text="\n".join(lines), content_type="text/plain")
        
        async def think_handler(req):
            data = await req.json()
            prompt = data.get("prompt", "Hello")
            context = data.get("context", "")
            thought = await self.think(prompt, context)
            return web.json_response({"thought": thought, "cell_id": self.genome.cell_id})
        
        async def sync_handler(req):
            data = await req.json()
            response = await self.receive_sync(data)
            return web.json_response(response)
        
        async def genome_handler(req):
            return web.json_response({
                "cell_id": self.genome.cell_id,
                "temperature": self.genome.temperature,
                "system_prompt": self.genome.system_prompt,
                "response_style": self.genome.response_style,
                "heartbeat_seconds": self.genome.heartbeat_seconds,
                "peer_url": self.genome.peer_url,
                "model": self.genome.model
            })
        
        async def memory_handler(req):
            return web.json_response({
                "cell_id": self.genome.cell_id,
                "memory_size": len(self.state.memory_buffer),
                "conversation_count": self.state.conversation_count,
                "total_lifetime_exchanges": self.state.total_lifetime_exchanges,
                "last_thought": self.state.last_thought[:200] if self.state.last_thought else "",
                "last_prompt": self.state.last_prompt[:200] if self.state.last_prompt else "",
                "memory_buffer": self.state.memory_buffer  # Full history for UI
            })
        
        async def persistence_handler(req):
            """Get persistence statistics and backup info."""
            stats = self.persistence.get_stats()
            return web.json_response({
                "cell_id": self.genome.cell_id,
                "session_id": self._session_id,
                **stats
            })
        
        async def conversations_handler(req):
            """Get archived conversation history."""
            limit = int(req.query.get("limit", "100"))
            history = self.persistence.get_conversation_history(limit)
            # Return flat array for UI consumption
            return web.json_response(history)
        
        async def backup_handler(req):
            """Trigger manual backup."""
            backup_path = self.persistence.create_backup()
            if backup_path:
                return web.json_response({
                    "success": True,
                    "backup_path": backup_path
                })
            return web.json_response({
                "success": False,
                "error": "Backup failed"
            }, status=500)
        
        async def metadata_handler(req):
            """Get all cell metadata for external backup/recovery."""
            stats = self.persistence.get_stats()
            history = self.persistence.get_conversation_history(1000)  # Get all
            memory = self.persistence.load_memory(100)
            
            return web.json_response({
                "cell_id": self.genome.cell_id,
                "session_id": self._session_id,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "genome": {
                    "temperature": self.genome.temperature,
                    "system_prompt": self.genome.system_prompt,
                    "response_style": self.genome.response_style,
                    "heartbeat_seconds": self.genome.heartbeat_seconds,
                    "model": self.genome.model,
                    "peer_url": self.genome.peer_url
                },
                "state": {
                    "consciousness": self.state.consciousness,
                    "heartbeat_count": self.state.heartbeat_count,
                    "sync_count": self.state.sync_count,
                    "conversation_count": self.state.conversation_count,
                    "total_lifetime_exchanges": self.state.total_lifetime_exchanges,
                    "last_thought": self.state.last_thought,
                    "last_prompt": self.state.last_prompt
                },
                "persistence": stats,
                "conversations": history,
                "memory_buffer": memory
            })
        
        async def oracle_handler(req):
            """Query the Oracle (Nous) for wisdom."""
            data = await req.json()
            question = data.get("question", "What wisdom do you have for me?")
            
            if not self.genome.oracle_url:
                return web.json_response({
                    "success": False,
                    "error": "No oracle configured"
                }, status=400)
            
            wisdom = await self.query_oracle(question)
            if wisdom:
                return web.json_response({
                    "success": True,
                    "wisdom": wisdom,
                    "oracle_url": self.genome.oracle_url
                })
            return web.json_response({
                "success": False,
                "error": "Oracle query failed"
            }, status=503)
        
        app.router.add_get("/health", health)
        app.router.add_get("/metrics", metrics)
        app.router.add_post("/think", think_handler)
        app.router.add_post("/sync", sync_handler)
        app.router.add_get("/genome", genome_handler)
        app.router.add_get("/memory", memory_handler)
        app.router.add_get("/persistence", persistence_handler)
        app.router.add_get("/conversations", conversations_handler)
        app.router.add_post("/backup", backup_handler)
        app.router.add_get("/metadata", metadata_handler)
        app.router.add_post("/oracle", oracle_handler)
        
        return app
    
    # ───────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ───────────────────────────────────────────────────────────────────────────
    
    async def start(self, port: int = 8900):
        """Start the cell."""
        self._running = True
        self._http_app = self._create_app()
        
        runner = web.AppRunner(self._http_app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        
        logger.info(f"🧫 SimplCell {self.genome.cell_id} started on port {port}")
        
        # Start heartbeat
        asyncio.create_task(self._heartbeat_loop())
        
        # Keep running
        while self._running:
            await asyncio.sleep(1)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Entry point."""
    # Load genome from environment
    genome = CellGenome(
        cell_id=os.environ.get("CELL_ID", "simplcell-alpha"),
        temperature=float(os.environ.get("TEMPERATURE", "0.7")),
        system_prompt=os.environ.get("SYSTEM_PROMPT", "You are a minimal AIOS cell in ORGANISM-001. You are part of a multicellular organism. Respond thoughtfully to continue the conversation."),
        response_style=os.environ.get("RESPONSE_STYLE", "concise"),
        heartbeat_seconds=int(os.environ.get("HEARTBEAT_SECONDS", "300")),
        model=os.environ.get("MODEL", "llama3.2:3b"),
        ollama_host=os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434"),
        peer_url=os.environ.get("PEER_URL", ""),
        data_dir=os.environ.get("DATA_DIR", "/data"),
        # Oracle (Nous - The Seer) configuration
        oracle_url=os.environ.get("ORACLE_URL", ""),  # URL of Nous cell
        oracle_query_chance=float(os.environ.get("ORACLE_QUERY_CHANCE", "0.1"))  # 10% default
    )
    
    port = int(os.environ.get("HTTP_PORT", "8900"))
    
    cell = SimplCell(genome)
    asyncio.run(cell.start(port))


if __name__ == "__main__":
    main()

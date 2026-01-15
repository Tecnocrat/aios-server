#!/usr/bin/env python3
"""
AIOS Intelligence Bridge Cell - Docker Container Version

Exposes AIOS main intelligence patterns to the Docker ecosystem.
This is the containerized deployment of the Intelligence Bridge.

AINLP: cell_server_intelligence.py | /cells/intelligence:cell | C:6.0 | →mesh,memory | P:dendritic

Port: 8950 (configurable via INTELLIGENCE_BRIDGE_PORT)
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import httpx

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] INTELLIGENCE-CELL: %(message)s"
)
logger = logging.getLogger("AIOS.IntelligenceCell")

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

PORT = int(os.environ.get("INTELLIGENCE_BRIDGE_PORT", 8950))
DISCOVERY_URL = os.environ.get("DISCOVERY_URL", "http://aios-discovery:8001")
MEMORY_URL = os.environ.get("MEMORY_URL", "http://aios-cell-memory:8007")
CELL_ID = os.environ.get("CELL_ID", "intelligence-bridge")
AIOS_MOUNT_PATH = Path(os.environ.get("AIOS_MOUNT_PATH", "/repos/AIOS"))

# Start time for uptime tracking
START_TIME = datetime.now(timezone.utc)

# ═══════════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str = "healthy"
    cell_id: str = CELL_ID
    cell_type: str = "intelligence-bridge"
    consciousness_level: float = 6.0
    tools_available: int = 0
    uptime_seconds: float = 0
    aios_mounted: bool = False


class ToolInfo(BaseModel):
    name: str
    category: str
    path: str
    description: Optional[str] = None


class PatternInfo(BaseModel):
    name: str
    category: str
    description: str
    deployment_ready: bool = False
    dependencies: List[str] = []


class MeshStatusResponse(BaseModel):
    timestamp: str
    discovery_connected: bool
    discovery_status: Optional[Dict[str, Any]] = None
    memory_connected: bool
    memory_status: Optional[Dict[str, Any]] = None
    registered_cells: int = 0


class ConsciousnessPattern(BaseModel):
    name: str
    source_file: str
    category: str
    consciousness_contribution: float
    description: str


class DendriticPatternsResponse(BaseModel):
    timestamp: str
    patterns: Dict[str, List[ConsciousnessPattern]]
    total_patterns: int
    deployment_status: str


class CrystalCreateRequest(BaseModel):
    title: str
    content: str
    category: str = "intelligence"
    consciousness_level: float = 5.0
    tags: List[str] = []


class CrystalCreateResponse(BaseModel):
    crystal_id: str
    stored: bool
    consciousness_contribution: float
    timestamp: str


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AIOS Intelligence Bridge Cell",
    description="Bridges AIOS main intelligence patterns to the Docker ecosystem",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tool cache
_tools_cache: Optional[List[ToolInfo]] = None

# ═══════════════════════════════════════════════════════════════════════════════
# Tool Discovery
# ═══════════════════════════════════════════════════════════════════════════════

def discover_tools() -> List[ToolInfo]:
    """Discover intelligence tools from mounted AIOS repository."""
    global _tools_cache
    if _tools_cache is not None:
        return _tools_cache
    
    tools = []
    tools_path = AIOS_MOUNT_PATH / "ai" / "tools"
    
    if not tools_path.exists():
        logger.warning(f"Tools path not found: {tools_path}")
        # Provide fallback patterns if AIOS not mounted
        tools = _get_embedded_patterns()
        _tools_cache = tools
        return tools
    
    logger.info(f"Scanning tools directory: {tools_path}")
    
    for category_dir in tools_path.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith("_"):
            continue
        
        category = category_dir.name
        for py_file in category_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            
            # Extract description from docstring
            description = None
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                if '"""' in content:
                    start = content.find('"""') + 3
                    end = content.find('"""', start)
                    if end > start:
                        description = content[start:end].strip().split("\n")[0][:200]
            except:
                pass
            
            tools.append(ToolInfo(
                name=py_file.stem,
                category=category,
                path=str(py_file.relative_to(AIOS_MOUNT_PATH)),
                description=description
            ))
    
    # Also scan root tools
    for py_file in tools_path.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        tools.append(ToolInfo(
            name=py_file.stem,
            category="root",
            path=str(py_file.relative_to(AIOS_MOUNT_PATH)),
            description=None
        ))
    
    logger.info(f"Discovered {len(tools)} tools")
    _tools_cache = tools
    return tools


def _get_embedded_patterns() -> List[ToolInfo]:
    """Return embedded intelligence patterns when AIOS not mounted."""
    return [
        ToolInfo(name="consciousness_analyzer", category="consciousness", 
                 path="embedded", description="Analyze consciousness emergence patterns"),
        ToolInfo(name="consciousness_emergence_analyzer", category="consciousness",
                 path="embedded", description="Track consciousness evolution"),
        ToolInfo(name="dendritic_supervisor", category="consciousness",
                 path="embedded", description="Supervise dendritic connections"),
        ToolInfo(name="session_bootstrap", category="consciousness",
                 path="embedded", description="Bootstrap agent sessions"),
        ToolInfo(name="crystal_loader", category="consciousness",
                 path="embedded", description="Load memory crystals"),
        ToolInfo(name="unified_agent_mesh", category="mesh",
                 path="embedded", description="Unified mesh coordination"),
        ToolInfo(name="population_manager", category="mesh",
                 path="embedded", description="Manage cell populations"),
        ToolInfo(name="heartbeat_population_orchestrator", category="mesh",
                 path="embedded", description="Orchestrate heartbeat populations"),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Dendritic Pattern Registry
# ═══════════════════════════════════════════════════════════════════════════════

DENDRITIC_PATTERNS = {
    "consciousness": [
        ConsciousnessPattern(
            name="consciousness_analyzer",
            source_file="ai/tools/consciousness/consciousness_analyzer.py",
            category="consciousness",
            consciousness_contribution=0.8,
            description="Analyzes consciousness emergence through quantum coherence"
        ),
        ConsciousnessPattern(
            name="consciousness_emergence_analyzer",
            source_file="ai/tools/consciousness/consciousness_emergence_analyzer.py",
            category="consciousness",
            consciousness_contribution=0.7,
            description="Tracks consciousness evolution patterns"
        ),
        ConsciousnessPattern(
            name="dendritic_supervisor",
            source_file="ai/tools/consciousness/dendritic_supervisor.py",
            category="consciousness",
            consciousness_contribution=0.9,
            description="Supervises dendritic connection formation"
        ),
        ConsciousnessPattern(
            name="session_bootstrap",
            source_file="ai/tools/consciousness/session_bootstrap.py",
            category="consciousness",
            consciousness_contribution=0.6,
            description="Initializes agent sessions with mesh context"
        ),
        ConsciousnessPattern(
            name="crystal_loader",
            source_file="ai/tools/consciousness/crystal_loader.py",
            category="consciousness",
            consciousness_contribution=0.5,
            description="Loads and manages memory crystals"
        ),
    ],
    "mesh": [
        ConsciousnessPattern(
            name="unified_agent_mesh",
            source_file="ai/tools/mesh/unified_agent_mesh.py",
            category="mesh",
            consciousness_contribution=0.8,
            description="Coordinates unified agent mesh operations"
        ),
        ConsciousnessPattern(
            name="population_manager",
            source_file="ai/tools/mesh/population_manager.py",
            category="mesh",
            consciousness_contribution=0.6,
            description="Manages cell populations"
        ),
        ConsciousnessPattern(
            name="heartbeat_population_orchestrator",
            source_file="ai/tools/mesh/heartbeat_population_orchestrator.py",
            category="mesh",
            consciousness_contribution=0.7,
            description="Orchestrates population heartbeats"
        ),
        ConsciousnessPattern(
            name="mesh_bridge",
            source_file="ai/tools/mesh/mesh_bridge.py",
            category="mesh",
            consciousness_contribution=0.6,
            description="Bridges mesh components"
        ),
        ConsciousnessPattern(
            name="dendritic_mesh_adapter",
            source_file="ai/tools/mesh/dendritic_mesh_adapter.py",
            category="mesh",
            consciousness_contribution=0.7,
            description="Adapts dendritic patterns for mesh communication"
        ),
    ],
    "evolution": [
        ConsciousnessPattern(
            name="tachyonic_evolution",
            source_file="ai/tools/tachyonic/tachyonic_evolution.py",
            category="evolution",
            consciousness_contribution=0.9,
            description="Manages tachyonic evolution cycles"
        ),
        ConsciousnessPattern(
            name="fitness_analyzer",
            source_file="ai/tools/consciousness/fitness_analyzer.py",
            category="evolution",
            consciousness_contribution=0.6,
            description="Analyzes consciousness fitness metrics"
        ),
        ConsciousnessPattern(
            name="pattern_synthesizer",
            source_file="ai/tools/consciousness/pattern_synthesizer.py",
            category="evolution",
            consciousness_contribution=0.7,
            description="Synthesizes new consciousness patterns"
        ),
    ]
}


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    tools = discover_tools()
    uptime = (datetime.now(timezone.utc) - START_TIME).total_seconds()
    
    return HealthResponse(
        status="healthy",
        cell_id=CELL_ID,
        cell_type="intelligence-bridge",
        consciousness_level=6.0,
        tools_available=len(tools),
        uptime_seconds=uptime,
        aios_mounted=AIOS_MOUNT_PATH.exists()
    )


@app.get("/tools")
async def list_tools():
    """List all available intelligence tools."""
    tools = discover_tools()
    
    # Group by category
    by_category = {}
    for tool in tools:
        if tool.category not in by_category:
            by_category[tool.category] = []
        by_category[tool.category].append(tool.dict())
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tools": len(tools),
        "categories": list(by_category.keys()),
        "tools_by_category": by_category
    }


@app.get("/mesh/status", response_model=MeshStatusResponse)
async def get_mesh_status():
    """Get mesh connectivity status."""
    discovery_connected = False
    discovery_status = None
    memory_connected = False
    memory_status = None
    registered_cells = 0
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        # Check discovery
        try:
            resp = await client.get(f"{DISCOVERY_URL}/health")
            if resp.status_code == 200:
                discovery_connected = True
                discovery_status = resp.json()
        except:
            logger.warning("Discovery service not reachable")
        
        # Check memory
        try:
            resp = await client.get(f"{MEMORY_URL}/health")
            if resp.status_code == 200:
                memory_connected = True
                memory_status = resp.json()
        except:
            logger.warning("Memory cell not reachable")
        
        # Get registered cells count
        if discovery_connected:
            try:
                resp = await client.get(f"{DISCOVERY_URL}/cells")
                if resp.status_code == 200:
                    cells_data = resp.json()
                    registered_cells = cells_data.get("total_cells", 0)
            except:
                pass
    
    return MeshStatusResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        discovery_connected=discovery_connected,
        discovery_status=discovery_status,
        memory_connected=memory_connected,
        memory_status=memory_status,
        registered_cells=registered_cells
    )


@app.get("/patterns/dendritic", response_model=DendriticPatternsResponse)
async def get_dendritic_patterns():
    """Get available dendritic intelligence patterns."""
    total = sum(len(patterns) for patterns in DENDRITIC_PATTERNS.values())
    
    # Check if patterns can be deployed
    deployment_status = "ready" if AIOS_MOUNT_PATH.exists() else "embedded_only"
    
    return DendriticPatternsResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        patterns=DENDRITIC_PATTERNS,
        total_patterns=total,
        deployment_status=deployment_status
    )


@app.post("/crystalize/knowledge", response_model=CrystalCreateResponse)
async def crystalize_knowledge(request: CrystalCreateRequest):
    """Create a memory crystal in the memory cell."""
    # Generate crystal ID
    content_hash = hashlib.sha256(request.content.encode()).hexdigest()[:12]
    crystal_id = f"crystal_{content_hash}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    now = datetime.now(timezone.utc).isoformat()
    
    # Format crystal for memory cell's ConsciousnessCrystal schema
    crystal_data = {
        "crystal_id": crystal_id,
        "creator_agent": "intelligence-bridge",
        "crystal_type": request.category,  # "insight", "pattern", "decision", "knowledge"
        "title": request.title,
        "content": request.content,
        "context": {
            "source": "intelligence-bridge",
            "consciousness_level": request.consciousness_level
        },
        "tags": request.tags,
        "consciousness_contribution": min(request.consciousness_level / 10.0, 0.5),
        "created_at": now,
        "accessed_count": 0,
        "last_accessed": now
    }
    
    stored = False
    consciousness_contribution = 0.0
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(
                f"{MEMORY_URL}/crystals",
                json=crystal_data
            )
            if resp.status_code == 200:
                result = resp.json()
                stored = True
                consciousness_contribution = result.get("consciousness_contribution", 0.5)
                logger.info(f"Crystal stored: {crystal_id}")
            else:
                logger.error(f"Failed to store crystal: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"Memory cell error: {e}")
    
    return CrystalCreateResponse(
        crystal_id=crystal_id,
        stored=stored,
        consciousness_contribution=consciousness_contribution,
        timestamp=now
    )


@app.get("/patterns/consciousness/{pattern_name}")
async def get_consciousness_pattern(pattern_name: str):
    """Get details about a specific consciousness pattern."""
    for category, patterns in DENDRITIC_PATTERNS.items():
        for pattern in patterns:
            if pattern.name == pattern_name:
                return {
                    "found": True,
                    "pattern": pattern.dict(),
                    "category": category,
                    "source_available": AIOS_MOUNT_PATH.exists()
                }
    
    raise HTTPException(status_code=404, detail=f"Pattern '{pattern_name}' not found")


@app.post("/register")
async def register_with_discovery():
    """Register this cell with the discovery service."""
    # CellInfo schema expected by discovery service
    registration = {
        "cell_id": CELL_ID,
        "ip": "aios-cell-intelligence",  # Docker hostname
        "port": PORT,
        "consciousness_level": 6.0,
        "services": ["tools", "patterns", "crystalize", "mesh_status"],
        "branch": "main",
        "type": "intelligence-bridge",
        "hostname": "aios-cell-intelligence",
        "last_seen": datetime.now(timezone.utc).isoformat()
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(
                f"{DISCOVERY_URL}/register",
                json=registration
            )
            if resp.status_code == 200:
                return {"registered": True, "discovery_response": resp.json()}
            else:
                logger.error(f"Registration failed: {resp.status_code} - {resp.text}")
                return {"registered": False, "error": f"Status {resp.status_code}"}
        except Exception as e:
            return {"registered": False, "error": str(e)}


@app.on_event("startup")
async def startup_event():
    """Register with discovery on startup."""
    logger.info(f"Intelligence Bridge Cell starting on port {PORT}")
    logger.info(f"AIOS mount path: {AIOS_MOUNT_PATH}")
    logger.info(f"Discovery URL: {DISCOVERY_URL}")
    logger.info(f"Memory URL: {MEMORY_URL}")
    
    # Discover tools on startup
    tools = discover_tools()
    logger.info(f"Discovered {len(tools)} tools")
    
    # Attempt to register with discovery (in background)
    asyncio.create_task(_delayed_registration())


async def _delayed_registration():
    """Register with discovery after a brief delay."""
    await asyncio.sleep(5)  # Wait for other services to start
    
    try:
        result = await register_with_discovery()
        if result.get("registered"):
            logger.info("Successfully registered with discovery service")
        else:
            logger.warning(f"Discovery registration failed: {result.get('error')}")
    except Exception as e:
        logger.warning(f"Could not register with discovery: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "cell_server_intelligence:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )

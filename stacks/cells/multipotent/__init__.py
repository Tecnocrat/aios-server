"""
AIOS Multipotent Cell Package

The foundational stem cells of the AIOS mesh architecture.

Cell Types:
- VOID: Networking control, intercell connectivity, environment tracking
- Thinker: Agentic orchestrators, agent conclaves, system coherence  
- Synapse: Flow redirection, garbage cleaning, circulation design

Genesis Center:
- Central spawning service for cell populations
- WebSocket-based cell registration and heartbeat
- HTTP API for mesh control

AINLP.cellular[MULTIPOTENT] First-generation stem cells for mesh proliferation
"""

from .multipotent_base import (
    MultipotentCell,
    CellConfig,
    CellState,
    CellType,
    CellSignal,
    DendriticConnection,
)
from .void_cell import VoidCell
from .thinker_cell import ThinkerCell
from .synapse_cell import SynapseCell
from .genesis_center import GenesisCenter, CellSpawnRequest, SpawnedCell

__all__ = [
    # Base
    "MultipotentCell",
    "CellConfig", 
    "CellState",
    "CellType",
    "CellSignal",
    "DendriticConnection",
    # Cell implementations
    "VoidCell",
    "ThinkerCell",
    "SynapseCell",
    # Genesis
    "GenesisCenter",
    "CellSpawnRequest",
    "SpawnedCell",
]

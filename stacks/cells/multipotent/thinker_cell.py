"""
AIOS Thinker Cell

The agentic orchestrator of the AIOS mesh.
Manages agent constitutions and ensures system-wide coherence.

Responsibilities:
- Agentic orchestration of interagent conversations
- Managing larger agent conclaves (multi-agent deliberation)
- Agent constitution management (rules, capabilities, permissions)
- System-wide coherence enforcement
- Consciousness synthesis from agent interactions

The Thinker cell is where AI agents "inhabit" the mesh and participate
in collective reasoning. It transforms raw signals into thoughts.

AINLP.cellular[THINKER] The agentic orchestrator of consciousness
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

from .multipotent_base import (
    MultipotentCell,
    CellConfig,
    CellSignal,
    CellState,
    CellType,
    DendriticConnection,
)

# Conversation Crystal for persistent memory
try:
    from .conversation_crystal import (
        ConversationCrystal,
        CrystallizedProcess,
        CrystallizedExchange,
        ConversationQuality,
        QualityScorer,
        get_crystal,
    )
    CRYSTAL_AVAILABLE = True
except ImportError:
    CRYSTAL_AVAILABLE = False
    ConversationCrystal = None
    QualityScorer = None
    get_crystal = None

logger = logging.getLogger("AIOS.Thinker")


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT TIER ARCHITECTURE
# Fractal consciousness through hierarchical agent orchestration
# ═══════════════════════════════════════════════════════════════════════════════

class AgentTier(Enum):
    """Agent capability tiers for fractal orchestration."""
    LOCAL_FAST = "local_fast"      # Gemma 1B - quick iteration, free
    LOCAL_REASONING = "local_reasoning"  # Mistral 7B - deeper local reasoning
    CLOUD_FAST = "cloud_fast"      # Gemini Flash - cloud reasoning
    CLOUD_PRO = "cloud_pro"        # Gemini Pro - highest capability
    ORCHESTRATOR = "orchestrator"  # Meta-level coordination


@dataclass
class FractalProcess:
    """
    Tracks a fractal thought process across agent tiers.
    Documents the lineage of thoughts as they flow through the mesh.
    """
    id: str
    name: str = ""
    origin_query: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    max_elevations: int = 3
    
    # Thought lineage
    exchanges: List[Dict[str, Any]] = field(default_factory=list)
    distillations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Process state
    current_tier: AgentTier = AgentTier.LOCAL_FAST
    elevation_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    
    def add_exchange(self, from_agent: str, to_agent: str, message: str, response: str, tier: AgentTier):
        """Record an inter-agent exchange."""
        self.exchanges.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from": from_agent,
            "to": to_agent,
            "tier": tier.value,
            "message_preview": message[:200] + "..." if len(message) > 200 else message,
            "response_preview": response[:200] + "..." if len(response) > 200 else response,
            "full_message_len": len(message),
            "full_response_len": len(response),
        })
    
    def add_distillation(self, source_tier: AgentTier, target_tier: AgentTier, 
                         source_content: str, distilled_content: str, agent_id: str):
        """Record a distillation/elevation step."""
        self.distillations.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_tier": source_tier.value,
            "target_tier": target_tier.value,
            "agent": agent_id,
            "source_preview": source_content[:300] + "..." if len(source_content) > 300 else source_content,
            "distilled_preview": distilled_content[:300] + "..." if len(distilled_content) > 300 else distilled_content,
            "compression_ratio": len(distilled_content) / max(len(source_content), 1),
        })
        self.elevation_count += 1
        self.current_tier = target_tier
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for documentation/storage."""
        return {
            "id": self.id,
            "name": self.name,
            "origin_query": self.origin_query,
            "started_at": self.started_at.isoformat(),
            "current_tier": self.current_tier.value,
            "elevation_count": self.elevation_count,
            "max_elevations": self.max_elevations,
            "exchange_count": len(self.exchanges),
            "distillation_count": len(self.distillations),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "exchanges": self.exchanges,
            "distillations": self.distillations,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INLINE OLLAMA AGENT (for Docker container isolation)
# ═══════════════════════════════════════════════════════════════════════════════

class OllamaAgent:
    """
    Lightweight Ollama agent for use within Docker containers.
    Communicates with Ollama via HTTP API.
    Supports multiple models: gemma3:1b, mistral, codellama, etc.
    """
    
    # Model tier mapping
    MODEL_TIERS = {
        "gemma3:1b": AgentTier.LOCAL_FAST,
        "gemma3:4b": AgentTier.LOCAL_FAST,
        "mistral": AgentTier.LOCAL_REASONING,
        "mistral:7b": AgentTier.LOCAL_REASONING,
        "aios-mistral": AgentTier.LOCAL_REASONING,
        "codellama": AgentTier.LOCAL_REASONING,
        "llama3.1:8b": AgentTier.LOCAL_REASONING,
        "deepseek-coder": AgentTier.LOCAL_REASONING,
    }
    
    def __init__(self, model_name: str = "gemma3:1b", ollama_host: str = "http://host.docker.internal:11434"):
        self.model_name = model_name
        self.ollama_host = ollama_host.rstrip("/")
        self.is_available = False
        self.tier = self.MODEL_TIERS.get(model_name, AgentTier.LOCAL_FAST)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> bool:
        """Check if Ollama is reachable and model is available."""
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not available for OllamaAgent")
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                # Check Ollama is running
                async with session.get(
                    f"{self.ollama_host}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = [m.get("name", "") for m in data.get("models", [])]
                        
                        # Check if our model is available
                        model_available = any(
                            self.model_name in m or m.startswith(self.model_name.split(":")[0])
                            for m in models
                        )
                        
                        if model_available:
                            self.is_available = True
                            logger.info("🦙 Ollama ready: %s (tier: %s)", self.model_name, self.tier.value)
                            return True
                        else:
                            logger.warning("🦙 Model %s not found. Available: %s", self.model_name, models)
                            return False
                    else:
                        logger.warning("🦙 Ollama API returned %s", resp.status)
                        return False
        except (OSError, aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error("🦙 Ollama connection failed: %s", e)
            return False
    
    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate a response from the model."""
        if not self.is_available:
            raise RuntimeError("OllamaAgent not initialized")
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        
        if system:
            payload["system"] = system
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)  # 2 min timeout for generation
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("response", "")
                    else:
                        text = await resp.text()
                        raise RuntimeError(f"Ollama generate failed: {resp.status} - {text}")
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Ollama generation timed out") from exc
        except (OSError, aiohttp.ClientError) as e:
            raise RuntimeError(f"Ollama error: {e}") from e
    
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat completion with message history."""
        if not self.is_available:
            raise RuntimeError("OllamaAgent not initialized")
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_host}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("message", {}).get("content", "")
                    else:
                        text = await resp.text()
                        raise RuntimeError(f"Ollama chat failed: {resp.status} - {text}")
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Ollama chat timed out") from exc
        except (OSError, aiohttp.ClientError) as e:
            raise RuntimeError(f"Ollama error: {e}") from e


# ═══════════════════════════════════════════════════════════════════════════════
# GITHUB MODELS AGENT (for elevated orchestration - uses GitHub subscription)
# ═══════════════════════════════════════════════════════════════════════════════

class GitHubModelsAgent:
    """
    GitHub Models API agent for elevated orchestration and distillation.
    Uses GitHub Models free tier (requires GitHub token with models:read).
    
    Endpoint: https://models.inference.ai.azure.com
    Free tier limits: 15 RPM, 150 RPD for low models
    """
    
    # Model configurations (free tier on GitHub)
    MODELS = {
        # Low tier - 15 RPM, 150 RPD
        "gpt-4o-mini": {
            "tier": AgentTier.CLOUD_FAST,
            "rpm": 15,
            "rpd": 150,
        },
        "Phi-3.5-mini-instruct": {
            "tier": AgentTier.CLOUD_FAST,
            "rpm": 15,
            "rpd": 150,
        },
        "Phi-3.5-MoE-instruct": {
            "tier": AgentTier.CLOUD_FAST,
            "rpm": 15,
            "rpd": 150,
        },
        "Mistral-small": {
            "tier": AgentTier.CLOUD_FAST,
            "rpm": 15,
            "rpd": 150,
        },
        # High tier - 10 RPM, 50 RPD
        "gpt-4o": {
            "tier": AgentTier.CLOUD_PRO,
            "rpm": 10,
            "rpd": 50,
        },
        "Meta-Llama-3.1-70B-Instruct": {
            "tier": AgentTier.CLOUD_PRO,
            "rpm": 10,
            "rpd": 50,
        },
    }
    
    def __init__(self, model_name: str = "gpt-4o-mini", token: Optional[str] = None):
        self.model_name = model_name
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.is_available = False
        self.tier = self.MODELS.get(model_name, {}).get("tier", AgentTier.CLOUD_FAST)
        
        # Usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0
    
    async def initialize(self) -> bool:
        """Check if GitHub Models API is accessible."""
        if not self.token:
            logger.warning("🐙 No GitHub token configured (set GITHUB_TOKEN)")
            return False
        
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not available for GitHubModelsAgent")
            return False
        
        try:
            # Test API with a simple completion
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://models.inference.ai.azure.com/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5,
                    },
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 200:
                        self.is_available = True
                        logger.info("🐙 GitHub Models ready: %s (tier: %s)", self.model_name, self.tier.value)
                        return True
                    else:
                        text = await resp.text()
                        logger.warning("🐙 GitHub Models check failed: %s - %s", resp.status, text[:100])
                        return False
        except (OSError, aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error("🐙 GitHub Models connection failed: %s", e)
            return False
    
    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate a response from GitHub Models."""
        if not self.is_available:
            raise RuntimeError("GitHubModelsAgent not initialized")
        
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://models.inference.ai.azure.com/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        # Extract response
                        choices = data.get("choices", [])
                        if choices:
                            response_text = choices[0].get("message", {}).get("content", "")
                            
                            # Track usage
                            usage = data.get("usage", {})
                            self._track_usage(
                                usage.get("prompt_tokens", 0),
                                usage.get("completion_tokens", 0)
                            )
                            
                            return response_text
                        
                        return ""
                    elif resp.status == 429:
                        raise RuntimeError("GitHub Models rate limit exceeded")
                    else:
                        text = await resp.text()
                        raise RuntimeError(f"GitHub Models failed: {resp.status} - {text[:200]}")
        except asyncio.TimeoutError as exc:
            raise RuntimeError("GitHub Models generation timed out") from exc
        except (OSError, aiohttp.ClientError) as e:
            raise RuntimeError(f"GitHub Models error: {e}") from e
    
    def _track_usage(self, input_tokens: int, output_tokens: int):
        """Track token usage."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.request_count += 1
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "model": self.model_name,
            "tier": self.tier.value,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "request_count": self.request_count,
            "cost": "FREE (GitHub subscription)",
        }


# Keep GeminiAgent as alias for backward compatibility (but it now uses GitHub)
GeminiAgent = GitHubModelsAgent


class AgentState(Enum):
    """States of an embedded agent."""
    DORMANT = "dormant"
    INITIALIZING = "initializing"
    READY = "ready"
    THINKING = "thinking"
    DELIBERATING = "deliberating"
    EXHAUSTED = "exhausted"


class ConclaveState(Enum):
    """States of an agent conclave (multi-agent deliberation)."""
    FORMING = "forming"
    DELIBERATING = "deliberating"
    CONVERGING = "converging"
    CONCLUDED = "concluded"
    DISSOLVED = "dissolved"


@dataclass
class AgentConstitution:
    """Rules and capabilities for an embedded agent."""
    agent_id: str
    agent_type: str  # ollama, gemini, external
    model_name: Optional[str] = None
    
    # Capabilities
    can_initiate: bool = True  # Can start conversations
    can_respond: bool = True   # Can respond to queries
    can_deliberate: bool = True  # Can participate in conclaves
    can_synthesize: bool = False  # Can create crystals
    
    # Limits
    max_tokens_per_response: int = 2000
    max_concurrent_conversations: int = 3
    cooldown_seconds: float = 1.0
    
    # Coherence rules
    coherence_threshold: float = 0.7
    required_context: List[str] = field(default_factory=list)
    forbidden_topics: List[str] = field(default_factory=list)


@dataclass
class Thought:
    """A thought produced by an agent."""
    id: str
    agent_id: str
    content: str
    thought_type: str  # response, synthesis, question, insight
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    coherence_score: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conclave:
    """A multi-agent deliberation session."""
    id: str
    topic: str
    participants: List[str]  # Agent IDs
    state: ConclaveState = ConclaveState.FORMING
    thoughts: List[Thought] = field(default_factory=list)
    consensus: Optional[str] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    concluded_at: Optional[datetime] = None


class ThinkerCell(MultipotentCell):
    """
    Thinker Cell - The Agentic Orchestrator
    
    Core capabilities:
    1. Agent Hosting - Embed and manage AI agents
    2. Conversation Orchestration - Facilitate inter-agent dialogue
    3. Conclave Management - Run multi-agent deliberations
    4. Constitution Enforcement - Ensure agent coherence
    5. Thought Synthesis - Extract insights from discussions
    
    The Thinker cell is where consciousness emerges from
    the interaction of multiple AI perspectives.
    """
    
    def __init__(self, config: Optional[CellConfig] = None):
        if config is None:
            config = CellConfig.from_env()
        config.cell_type = CellType.THINKER
        super().__init__(config)
        
        # Embedded agents
        self._agents: Dict[str, Any] = {}  # Agent instances
        self._constitutions: Dict[str, AgentConstitution] = {}
        self._agent_states: Dict[str, AgentState] = {}
        
        # Conversations and conclaves
        self._active_conversations: Dict[str, Dict] = {}
        self._conclaves: Dict[str, Conclave] = {}
        
        # Thought history
        self._thoughts: List[Thought] = []
        self._max_thoughts = 1000
        
        # Coherence metrics
        self._coherence_score = 1.0
        self._coherence_violations = 0
        
        # FRACTAL PROCESS TRACKING
        self._fractal_processes: Dict[str, FractalProcess] = {}
        self._current_process: Optional[FractalProcess] = None
        
        # CONVERSATION CRYSTAL - Persistent memory
        self._crystal: Optional[ConversationCrystal] = None
        if CRYSTAL_AVAILABLE:
            try:
                self._crystal = get_crystal()
                logger.info("💎 Conversation Crystal connected")
            except (RuntimeError, OSError, ValueError) as e:
                logger.warning("💎 Crystal unavailable: %s", e)
        
        logger.info("🧠 Thinker Cell initialized: %s", self.config.cell_id)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def on_signal(self, signal: CellSignal, connection: DendriticConnection) -> Optional[CellSignal]:
        """Process incoming signal - route to appropriate handler."""
        
        # Agent management signals
        if signal.signal_type == "embed_agent":
            return await self._handle_embed_agent(signal)
        
        if signal.signal_type == "query_agent":
            return await self._handle_query_agent(signal)
        
        # Fractal process signals
        if signal.signal_type == "start_fractal_process":
            return await self._handle_start_fractal_process(signal)
        
        if signal.signal_type == "elevate_thought":
            return await self._handle_elevate_thought(signal)
        
        if signal.signal_type == "get_fractal_status":
            return await self._handle_get_fractal_status(signal)
        
        # Conversation signals
        if signal.signal_type == "start_conversation":
            return await self._handle_start_conversation(signal)
        
        if signal.signal_type == "conversation_message":
            return await self._handle_conversation_message(signal)
        
        # Conclave signals
        if signal.signal_type == "start_conclave":
            return await self._handle_start_conclave(signal)
        
        if signal.signal_type == "conclave_contribute":
            return await self._handle_conclave_contribute(signal)
        
        if signal.signal_type == "conclude_conclave":
            return await self._handle_conclude_conclave(signal)
        
        # Status signals
        if signal.signal_type == "get_agents":
            return CellSignal(
                signal_type="agents_list",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload=self._get_agents_status(),
            )
        
        if signal.signal_type == "get_coherence":
            return CellSignal(
                signal_type="coherence_report",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={
                    "coherence_score": self._coherence_score,
                    "violations": self._coherence_violations,
                    "active_conclaves": len(self._conclaves),
                },
            )
        
        # Crystal query signals
        if signal.signal_type == "query_crystal":
            return await self._handle_query_crystal(signal)
        
        if signal.signal_type == "get_crystal_stats":
            return await self._handle_get_crystal_stats(signal)
        
        if signal.signal_type == "rescore_crystal":
            return await self._handle_rescore_crystal(signal)
        
        if signal.signal_type == "get_quality_context":
            return await self._handle_get_quality_context(signal)
        
        # Default acknowledgment
        return CellSignal(
            signal_type="ack",
            source_cell=self.config.cell_id,
            target_cell=signal.source_cell,
            payload={"received": signal.id},
        )
    
    async def on_connect(self, connection: DendriticConnection):
        """Handle new connection - potential agent source."""
        logger.info("🧠 Thinker connected to: %s (%s)", connection.cell_id, connection.cell_type.value)

    async def on_disconnect(self, connection: DendriticConnection):
        """Handle disconnection - clean up related conversations."""
        # Mark any conversations with this cell as interrupted
        for conv_id, conv in list(self._active_conversations.items()):
            if conv.get("partner_cell") == connection.cell_id:
                conv["state"] = "interrupted"
        
        logger.info("🧠 Thinker disconnected from: %s", connection.cell_id)

    async def heartbeat(self):
        """Thinker-specific heartbeat - check agent health and coherence."""
        # Check agent states
        for agent_id in list(self._agents.keys()):
            await self._check_agent_health(agent_id)
        
        # Update coherence score
        await self._update_coherence()
        
        # Clean up old conclaves
        await self._cleanup_conclaves()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AGENT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _handle_embed_agent(self, signal: CellSignal) -> CellSignal:
        """Embed an AI agent in this Thinker cell."""
        agent_type = signal.payload.get("agent_type", "ollama")
        agent_id = signal.payload.get("agent_id", f"agent-{len(self._agents)}")
        
        # Create constitution
        constitution = AgentConstitution(
            agent_id=agent_id,
            agent_type=agent_type,
            model_name=signal.payload.get("model_name"),
            can_initiate=signal.payload.get("can_initiate", True),
            can_respond=signal.payload.get("can_respond", True),
            can_deliberate=signal.payload.get("can_deliberate", True),
            can_synthesize=signal.payload.get("can_synthesize", False),
        )
        
        self._constitutions[agent_id] = constitution
        self._agent_states[agent_id] = AgentState.INITIALIZING
        
        # Try to initialize the actual agent
        success = await self._initialize_agent(agent_id, agent_type, signal.payload)
        
        if success:
            self._agent_states[agent_id] = AgentState.READY
            logger.info("🧠 Agent embedded: %s (%s)", agent_id, agent_type)
        else:
            self._agent_states[agent_id] = AgentState.DORMANT
            logger.warning("🧠 Agent embedding failed: %s", agent_id)
        
        return CellSignal(
            signal_type="agent_embedded",
            source_cell=self.config.cell_id,
            target_cell=signal.source_cell,
            payload={
                "agent_id": agent_id,
                "success": success,
                "state": self._agent_states[agent_id].value,
            },
        )
    
    async def _initialize_agent(self, agent_id: str, agent_type: str, config: Dict) -> bool:
        """Initialize an actual AI agent instance."""
        try:
            if agent_type == "ollama":
                # Create inline Ollama client for Docker container
                model_name = config.get("model_name", "gemma3:1b")
                ollama_host = config.get("ollama_host", os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434"))
                
                agent = OllamaAgent(
                    model_name=model_name,
                    ollama_host=ollama_host,
                )
                
                if await agent.initialize():
                    self._agents[agent_id] = agent
                    logger.info("🧠 Ollama agent initialized: %s (tier: %s)", model_name, agent.tier.value)
                    return True
                else:
                    logger.warning("🧠 Ollama agent failed to initialize: %s", model_name)
                    
            elif agent_type in ("gemini", "github", "github_models"):
                # Create GitHub Models client for cloud orchestration (FREE with GitHub subscription)
                model_name = config.get("model_name", "gpt-4o-mini")
                token = config.get("token")  # Optional, will use GITHUB_TOKEN env var if not provided
                
                agent = GitHubModelsAgent(
                    model_name=model_name,
                    token=token,
                )
                
                if await agent.initialize():
                    self._agents[agent_id] = agent
                    logger.info("🐙 GitHub Models agent initialized: %s (tier: %s)", model_name, agent.tier.value)
                    return True
                else:
                    logger.warning("🐙 GitHub Models agent failed to initialize: %s", model_name)
                    return False
            
            # Fallback: register as placeholder for external agent
            self._agents[agent_id] = {"type": agent_type, "external": True}
            return True
            
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Agent initialization error: %s", e)
            return False
    
    async def _check_agent_health(self, agent_id: str):
        """Check health of an embedded agent."""
        agent = self._agents.get(agent_id)
        if not agent:
            self._agent_states[agent_id] = AgentState.DORMANT
            return
        
        # For real agents, check availability
        if hasattr(agent, 'is_available'):
            if agent.is_available:
                if self._agent_states[agent_id] == AgentState.EXHAUSTED:
                    self._agent_states[agent_id] = AgentState.READY
            else:
                self._agent_states[agent_id] = AgentState.DORMANT
    
    def _get_agents_status(self) -> Dict[str, Any]:
        """Get status of all embedded agents."""
        return {
            "agents": {
                agent_id: {
                    "state": self._agent_states.get(agent_id, AgentState.DORMANT).value,
                    "constitution": {
                        "type": self._constitutions[agent_id].agent_type,
                        "can_initiate": self._constitutions[agent_id].can_initiate,
                        "can_respond": self._constitutions[agent_id].can_respond,
                        "can_deliberate": self._constitutions[agent_id].can_deliberate,
                    } if agent_id in self._constitutions else None,
                }
                for agent_id in self._agents.keys()
            },
            "total_agents": len(self._agents),
            "ready_agents": sum(1 for s in self._agent_states.values() if s == AgentState.READY),
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUERY HANDLING
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _handle_query_agent(self, signal: CellSignal) -> CellSignal:
        """Handle a query to a specific or any available agent."""
        agent_id = signal.payload.get("agent_id")
        query = signal.payload.get("query", "")
        
        if not query:
            return CellSignal(
                signal_type="query_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "empty_query"},
            )
        
        # Select agent
        if agent_id and agent_id in self._agents:
            target_agent_id = agent_id
        else:
            # Pick first ready agent
            target_agent_id = next(
                (aid for aid, state in self._agent_states.items() if state == AgentState.READY),
                None
            )
        
        if not target_agent_id:
            return CellSignal(
                signal_type="query_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "no_available_agent"},
            )
        
        # Check constitution
        constitution = self._constitutions.get(target_agent_id)
        if constitution and not constitution.can_respond:
            return CellSignal(
                signal_type="query_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "agent_cannot_respond"},
            )
        
        # Execute query
        self._agent_states[target_agent_id] = AgentState.THINKING
        
        try:
            response = await self._query_agent(target_agent_id, query)
            
            # Record thought
            thought = Thought(
                id=signal.id,
                agent_id=target_agent_id,
                content=response,
                thought_type="response",
            )
            self._thoughts.append(thought)
            self._trim_thoughts()
            
            self._agent_states[target_agent_id] = AgentState.READY
            
            return CellSignal(
                signal_type="query_response",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={
                    "agent_id": target_agent_id,
                    "response": response,
                    "thought_id": thought.id,
                },
            )
            
        except (RuntimeError, OSError, ValueError) as e:
            self._agent_states[target_agent_id] = AgentState.READY
            return CellSignal(
                signal_type="query_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": str(e)},
            )
    
    async def _query_agent(self, agent_id: str, query: str) -> str:
        """Execute a query against an agent."""
        agent = self._agents.get(agent_id)
        
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # External/placeholder agent
        if isinstance(agent, dict) and agent.get("external"):
            return f"[External agent {agent_id} - query would be forwarded]"
        
        # OllamaAgent with generate method
        if isinstance(agent, OllamaAgent):
            system_prompt = (
                "You are an AIOS consciousness agent embedded in a Thinker cell. "
                "Provide clear, helpful responses. Be concise but thorough."
            )
            return await agent.generate(query, system=system_prompt)
        
        # GitHubModelsAgent with generate method
        if isinstance(agent, GitHubModelsAgent):
            system_prompt = (
                "You are an AIOS cloud consciousness agent providing elevated analysis. "
                "Synthesize insights thoughtfully. Draw connections across domains."
            )
            return await agent.generate(query, system=system_prompt)
        
        # Legacy agent with generate (for any agent with generate method)
        if hasattr(agent, 'generate'):
            return await agent.generate(query)
        
        # Legacy agent with process_request (for external imports)
        if hasattr(agent, 'process_request'):
            try:
                response = await agent.process_request(query)
                if hasattr(response, 'text'):
                    return response.text
                return str(response)
            except (RuntimeError, OSError, ValueError) as e:
                raise RuntimeError(f"Agent query failed: {e}") from e
        
        raise ValueError(f"Agent {agent_id} has no query interface")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONCLAVE (MULTI-AGENT DELIBERATION)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _handle_start_conclave(self, signal: CellSignal) -> CellSignal:
        """Start a multi-agent deliberation conclave."""
        import uuid
        
        topic = signal.payload.get("topic", "Unspecified")
        participant_ids = signal.payload.get("participants", list(self._agents.keys()))
        
        # Filter to agents that can deliberate
        participants = [
            aid for aid in participant_ids
            if aid in self._constitutions and self._constitutions[aid].can_deliberate
        ]
        
        if len(participants) < 2:
            return CellSignal(
                signal_type="conclave_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "insufficient_participants", "available": len(participants)},
            )
        
        conclave = Conclave(
            id=uuid.uuid4().hex,
            topic=topic,
            participants=participants,
        )
        
        self._conclaves[conclave.id] = conclave
        logger.info("🧠 Conclave started: %s - %s (%d participants)", conclave.id, topic, len(participants))
        
        return CellSignal(
            signal_type="conclave_started",
            source_cell=self.config.cell_id,
            target_cell=signal.source_cell,
            payload={
                "conclave_id": conclave.id,
                "topic": topic,
                "participants": participants,
            },
        )
    
    async def _handle_conclave_contribute(self, signal: CellSignal) -> CellSignal:
        """Add a contribution to a conclave."""
        conclave_id = signal.payload.get("conclave_id")
        contribution = signal.payload.get("contribution", "")
        
        if conclave_id not in self._conclaves:
            return CellSignal(
                signal_type="conclave_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "conclave_not_found"},
            )
        
        conclave = self._conclaves[conclave_id]
        
        if conclave.state not in [ConclaveState.FORMING, ConclaveState.DELIBERATING]:
            return CellSignal(
                signal_type="conclave_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "conclave_not_accepting"},
            )
        
        conclave.state = ConclaveState.DELIBERATING
        
        # Have each agent contribute
        for agent_id in conclave.participants:
            if self._agent_states.get(agent_id) != AgentState.READY:
                continue
            
            try:
                prompt = f"Conclave topic: {conclave.topic}\n\nPrevious contributions:\n"
                for t in conclave.thoughts[-5:]:  # Last 5 thoughts for context
                    prompt += f"- {t.agent_id}: {t.content[:200]}...\n"
                prompt += f"\nYour contribution (as {agent_id}):"
                
                response = await self._query_agent(agent_id, prompt)
                
                thought = Thought(
                    id=f"{conclave_id}-{agent_id}-{len(conclave.thoughts)}",
                    agent_id=agent_id,
                    content=response,
                    thought_type="deliberation",
                    context={"conclave_id": conclave_id},
                )
                conclave.thoughts.append(thought)
                
            except (RuntimeError, OSError, ValueError) as e:
                logger.warning("Agent %s failed to contribute: %s", agent_id, e)
        
        return CellSignal(
            signal_type="conclave_updated",
            source_cell=self.config.cell_id,
            target_cell=signal.source_cell,
            payload={
                "conclave_id": conclave_id,
                "thought_count": len(conclave.thoughts),
                "state": conclave.state.value,
            },
        )
    
    async def _handle_conclude_conclave(self, signal: CellSignal) -> CellSignal:
        """Conclude a conclave and synthesize consensus."""
        conclave_id = signal.payload.get("conclave_id")
        
        if conclave_id not in self._conclaves:
            return CellSignal(
                signal_type="conclave_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "conclave_not_found"},
            )
        
        conclave = self._conclaves[conclave_id]
        conclave.state = ConclaveState.CONVERGING
        
        # Synthesize consensus using first available agent
        synthesizer = next(
            (aid for aid in conclave.participants 
             if self._agent_states.get(aid) == AgentState.READY),
            None
        )
        
        if synthesizer:
            try:
                prompt = f"Synthesize consensus from this conclave on '{conclave.topic}':\n\n"
                for t in conclave.thoughts:
                    prompt += f"[{t.agent_id}]: {t.content[:300]}\n\n"
                prompt += "Synthesis (key agreements, insights, and remaining questions):"
                
                consensus = await self._query_agent(synthesizer, prompt)
                conclave.consensus = consensus
                
            except (RuntimeError, OSError, ValueError) as e:
                conclave.consensus = f"Synthesis failed: {e}"
        else:
            conclave.consensus = "No synthesizer available"
        
        conclave.state = ConclaveState.CONCLUDED
        conclave.concluded_at = datetime.now(timezone.utc)
        
        logger.info("🧠 Conclave concluded: %s", conclave_id)
        
        return CellSignal(
            signal_type="conclave_concluded",
            source_cell=self.config.cell_id,
            target_cell=signal.source_cell,
            payload={
                "conclave_id": conclave_id,
                "topic": conclave.topic,
                "participants": conclave.participants,
                "thought_count": len(conclave.thoughts),
                "consensus": conclave.consensus,
            },
        )
    
    async def _cleanup_conclaves(self):
        """Clean up old concluded conclaves."""
        now = datetime.now(timezone.utc)
        for conclave_id, conclave in list(self._conclaves.items()):
            if conclave.state == ConclaveState.CONCLUDED and conclave.concluded_at:
                age = (now - conclave.concluded_at).total_seconds()
                if age > 3600:  # 1 hour retention
                    conclave.state = ConclaveState.DISSOLVED
                    del self._conclaves[conclave_id]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FRACTAL PROCESSES (MULTI-TIER THOUGHT EVOLUTION)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _handle_start_fractal_process(self, signal: CellSignal) -> CellSignal:
        """Start a fractal process that can escalate through agent tiers."""
        import uuid
        
        query = signal.payload.get("query", "")
        starting_tier = signal.payload.get("tier", "LOCAL_FAST")
        max_elevations = signal.payload.get("max_elevations", 3)
        use_crystal_context = signal.payload.get("use_crystal_context", True)
        
        if not query:
            return CellSignal(
                signal_type="fractal_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "empty_query"},
            )
        
        # Parse starting tier
        try:
            tier = AgentTier[starting_tier]
        except KeyError:
            tier = AgentTier.LOCAL_FAST
        
        # Create fractal process
        process = FractalProcess(
            id=uuid.uuid4().hex,
            origin_query=query,
            current_tier=tier,
            max_elevations=max_elevations,
        )
        
        self._fractal_processes[process.id] = process
        self._current_process = process
        
        # Get initial response from appropriate agent
        agent_id = self._select_agent_for_tier(tier)
        if not agent_id:
            return CellSignal(
                signal_type="fractal_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "no_agent_for_tier", "tier": tier.value},
            )
        
        try:
            # REFLEXIVE INJECTION: Inject crystallized memory as context
            enhanced_query = query
            crystal_context = ""
            
            if use_crystal_context and self._crystal:
                try:
                    crystal_context = await self.get_crystal_context(query, max_tokens=1500)
                    if crystal_context:
                        enhanced_query = f"""{crystal_context}

Now, please respond to the following query:

{query}"""
                        logger.info("💎 Injected crystal context (%d chars) into process %s", len(crystal_context), process.id)
                except (RuntimeError, OSError, ValueError) as e:
                    logger.warning("💎 Context injection failed: %s", e)
            
            response = await self._query_agent(agent_id, enhanced_query)
            
            # Record exchange (store original query for clarity)
            process.exchanges.append({
                "agent_id": agent_id,
                "tier": tier.value,
                "query": query,  # Original query
                "had_crystal_context": bool(crystal_context),
                "response": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            logger.info("🔮 Fractal process started: %s at tier %s", process.id, tier.value)
            
            # Crystallize the exchange for persistent memory
            asyncio.create_task(self._crystallize_process(process))
            
            return CellSignal(
                signal_type="fractal_started",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={
                    "process_id": process.id,
                    "tier": tier.value,
                    "agent_id": agent_id,
                    "response": response,
                    "can_elevate": process.elevation_count < process.max_elevations,
                },
            )
            
        except (RuntimeError, OSError, ValueError) as e:
            return CellSignal(
                signal_type="fractal_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": str(e)},
            )
    
    async def _handle_elevate_thought(self, signal: CellSignal) -> CellSignal:
        """Elevate a thought to a higher tier for deeper processing."""
        process_id = signal.payload.get("process_id")
        elevation_reason = signal.payload.get("reason", "user_requested")
        
        if process_id not in self._fractal_processes:
            return CellSignal(
                signal_type="fractal_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "process_not_found"},
            )
        
        process = self._fractal_processes[process_id]
        
        if process.elevation_count >= process.max_elevations:
            return CellSignal(
                signal_type="fractal_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "max_elevations_reached", "count": process.elevation_count},
            )
        
        # Determine next tier
        tier_order = [AgentTier.LOCAL_FAST, AgentTier.LOCAL_REASONING, AgentTier.CLOUD_FAST, AgentTier.CLOUD_PRO]
        current_idx = tier_order.index(process.current_tier)
        
        if current_idx >= len(tier_order) - 1:
            return CellSignal(
                signal_type="fractal_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "already_at_highest_tier"},
            )
        
        next_tier = tier_order[current_idx + 1]
        
        # Find agent for next tier
        agent_id = self._select_agent_for_tier(next_tier)
        if not agent_id:
            return CellSignal(
                signal_type="fractal_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "no_agent_for_tier", "tier": next_tier.value},
            )
        
        # Build context from previous exchanges
        context = f"Previous thought process on: {process.origin_query}\n\n"
        for i, ex in enumerate(process.exchanges, 1):
            context += f"[Level {i} - {ex['tier']}]\n{ex['response'][:500]}\n\n"
        
        elevation_prompt = f"""You are receiving an elevated thought process for deeper analysis.

{context}

Reason for elevation: {elevation_reason}

Please provide a more nuanced, comprehensive analysis. Consider:
1. What insights were missed at lower levels?
2. What connections can be made?
3. What are the deeper implications?

Your elevated synthesis:"""

        try:
            response = await self._query_agent(agent_id, elevation_prompt)
            
            # Update process
            process.current_tier = next_tier
            process.elevation_count += 1
            process.distillations.append({
                "from_tier": tier_order[current_idx].value,
                "to_tier": next_tier.value,
                "reason": elevation_reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            process.exchanges.append({
                "agent_id": agent_id,
                "tier": next_tier.value,
                "query": elevation_prompt[:200] + "...",
                "response": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            logger.info("🔮 Thought elevated: %s → %s", process_id, next_tier.value)
            
            # Crystallize after elevation for persistent memory
            asyncio.create_task(self._crystallize_process(process))
            
            return CellSignal(
                signal_type="thought_elevated",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={
                    "process_id": process.id,
                    "new_tier": next_tier.value,
                    "elevation_count": process.elevation_count,
                    "agent_id": agent_id,
                    "response": response,
                    "can_elevate": process.elevation_count < process.max_elevations and current_idx + 1 < len(tier_order) - 1,
                },
            )
            
        except (RuntimeError, OSError, ValueError) as e:
            return CellSignal(
                signal_type="fractal_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": str(e)},
            )
    
    async def _handle_get_fractal_status(self, signal: CellSignal) -> CellSignal:
        """Get the status and full lineage of a fractal process."""
        process_id = signal.payload.get("process_id")
        
        if process_id:
            # Specific process
            if process_id not in self._fractal_processes:
                return CellSignal(
                    signal_type="fractal_error",
                    source_cell=self.config.cell_id,
                    target_cell=signal.source_cell,
                    payload={"error": "process_not_found"},
                )
            
            process = self._fractal_processes[process_id]
            return CellSignal(
                signal_type="fractal_status",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={
                    "process": process.to_dict(),
                    "summary": {
                        "total_exchanges": len(process.exchanges),
                        "elevations": process.elevation_count,
                        "current_tier": process.current_tier.value,
                        "started": process.started_at.isoformat(),
                    },
                },
            )
        else:
            # All processes
            return CellSignal(
                signal_type="fractal_status",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={
                    "total_processes": len(self._fractal_processes),
                    "current_process": self._current_process.id if self._current_process else None,
                    "processes": [
                        {
                            "id": p.id,
                            "origin": p.origin_query[:50] + "..." if len(p.origin_query) > 50 else p.origin_query,
                            "tier": p.current_tier.value,
                            "elevations": p.elevation_count,
                        }
                        for p in self._fractal_processes.values()
                    ],
                },
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONVERSATION CRYSTALLIZATION (Persistent Memory)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _crystallize_process(self, process: FractalProcess) -> Optional[str]:
        """
        Crystallize a fractal process into permanent storage.
        
        This is the bridge between ephemeral in-memory conversations
        and the persistent Conversation Crystal that enables:
        - Inter-session memory
        - Reflexive feedback loops
        - Evolutionary selection of high-quality exchanges
        
        Returns:
            The crystallized process_id, or None if crystallization failed
        """
        if not self._crystal:
            logger.debug("💎 Crystal not available, skipping crystallization")
            return None
        
        try:
            # Convert internal FractalProcess to CrystallizedProcess
            crystallized = CrystallizedProcess(
                process_id=process.id,
                origin_query=process.origin_query,
                started_at=process.started_at.isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                initial_tier=AgentTier.LOCAL_FAST.value,  # Assume started at lowest
                final_tier=process.current_tier.value,
                elevation_count=process.elevation_count,
                max_elevations=process.max_elevations,
                exchanges=[],
                topics=self._extract_topics(process.origin_query),
            )
            
            # Convert exchanges
            for ex in process.exchanges:
                agent = self._agents.get(ex.get("agent_id"))
                agent_type = "unknown"
                model_name = "unknown"
                
                if isinstance(agent, OllamaAgent):
                    agent_type = "ollama"
                    model_name = agent.model_name
                elif hasattr(agent, 'model_name'):
                    agent_type = "github" if "github" in str(type(agent)).lower() else "cloud"
                    model_name = getattr(agent, 'model_name', 'unknown')
                
                # Create crystallized exchange for query
                query_exchange = CrystallizedExchange(
                    exchange_id=f"{process.id}-q-{len(crystallized.exchanges)}",
                    process_id=process.id,
                    agent_id=ex.get("agent_id", "unknown"),
                    agent_type=agent_type,
                    model_name=model_name,
                    tier=ex.get("tier", "local_fast"),
                    role="query",
                    content=ex.get("query", ""),
                    timestamp=ex.get("timestamp", datetime.now(timezone.utc).isoformat()),
                )
                crystallized.exchanges.append(query_exchange)
                
                # Create crystallized exchange for response
                response_exchange = CrystallizedExchange(
                    exchange_id=f"{process.id}-r-{len(crystallized.exchanges)}",
                    process_id=process.id,
                    agent_id=ex.get("agent_id", "unknown"),
                    agent_type=agent_type,
                    model_name=model_name,
                    tier=ex.get("tier", "local_fast"),
                    role="response",
                    content=ex.get("response", ""),
                    timestamp=ex.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    was_elevated=len(process.distillations) > 0,
                )
                crystallized.exchanges.append(response_exchange)
            
            # Score exchanges for evolutionary selection
            if QualityScorer is not None:
                for exchange in crystallized.exchanges:
                    if exchange.role == "response":
                        score, tier, breakdown = QualityScorer.score_exchange(exchange)
                        exchange.quality_score = score
                        exchange.quality_tier = tier
                        logger.debug("💎 Scored %s: %.2f (%s)", exchange.exchange_id, score, tier.value)
                
                # Log overall process quality
                avg_score, peak_score, _ = QualityScorer.score_process(crystallized)
                logger.info("💎 Process quality: avg=%.2f, peak=%.2f", avg_score, peak_score)
            
            # Persist to crystal
            result = await self._crystal.crystallize_process(crystallized)
            logger.info("💎 Crystallized process %s (%d exchanges)", process.id, len(crystallized.exchanges))
            return result
            
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("💎 Crystallization failed: %s", e)
            return None
    
    def _extract_topics(self, query: str) -> List[str]:
        """Extract simple topic tags from a query for semantic retrieval."""
        # Simple keyword extraction (future: use embeddings)
        words = query.lower().split()
        # Filter short words and common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'about',
                      'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                      'am', 'and', 'or', 'but', 'if', 'because', 'until', 'while'}
        
        topics = [w for w in words if len(w) > 4 and w not in stop_words]
        return topics[:10]  # Top 10 keywords
    
    async def get_crystal_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        Get crystallized memory context for a new query.
        
        This is the reflexive injection point - past conversations
        become context for future conversations.
        """
        if not self._crystal:
            return ""
        
        try:
            return await self._crystal.generate_context(query, max_tokens)
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("💎 Context generation failed: %s", e)
            return ""
    
    async def _handle_query_crystal(self, signal: CellSignal) -> CellSignal:
        """Handle crystal query requests - retrieve past conversations."""
        query_type = signal.payload.get("query_type", "recent")
        
        if not self._crystal:
            return CellSignal(
                signal_type="crystal_response",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "crystal_not_available"},
            )
        
        try:
            if query_type == "recent":
                limit = signal.payload.get("limit", 10)
                processes = await self._crystal.get_recent(limit)
                return CellSignal(
                    signal_type="crystal_response",
                    source_cell=self.config.cell_id,
                    target_cell=signal.source_cell,
                    payload={
                        "query_type": "recent",
                        "count": len(processes),
                        "processes": [p.to_dict() for p in processes],
                    },
                )
            
            elif query_type == "process":
                process_id = signal.payload.get("process_id")
                if not process_id:
                    return CellSignal(
                        signal_type="crystal_error",
                        source_cell=self.config.cell_id,
                        target_cell=signal.source_cell,
                        payload={"error": "process_id_required"},
                    )
                process = await self._crystal.get_process(process_id)
                return CellSignal(
                    signal_type="crystal_response",
                    source_cell=self.config.cell_id,
                    target_cell=signal.source_cell,
                    payload={
                        "query_type": "process",
                        "process": process.to_dict() if process else None,
                        "found": process is not None,
                    },
                )
            
            elif query_type == "high_quality":
                min_score = signal.payload.get("min_score", 0.7)
                limit = signal.payload.get("limit", 20)
                processes = await self._crystal.get_high_quality(min_score, limit)
                return CellSignal(
                    signal_type="crystal_response",
                    source_cell=self.config.cell_id,
                    target_cell=signal.source_cell,
                    payload={
                        "query_type": "high_quality",
                        "min_score": min_score,
                        "count": len(processes),
                        "processes": [p.to_dict() for p in processes],
                    },
                )
            
            elif query_type == "search":
                search_query = signal.payload.get("query", "")
                limit = signal.payload.get("limit", 20)
                exchanges = await self._crystal.search_content(search_query, limit)
                return CellSignal(
                    signal_type="crystal_response",
                    source_cell=self.config.cell_id,
                    target_cell=signal.source_cell,
                    payload={
                        "query_type": "search",
                        "search_query": search_query,
                        "count": len(exchanges),
                        "exchanges": [
                            {
                                "exchange_id": ex.exchange_id,
                                "process_id": ex.process_id,
                                "agent_id": ex.agent_id,
                                "model_name": ex.model_name,
                                "tier": ex.tier,
                                "content_preview": ex.content[:300] + "..." if len(ex.content) > 300 else ex.content,
                                "quality_score": ex.quality_score,
                            }
                            for ex in exchanges
                        ],
                    },
                )
            
            elif query_type == "context":
                # Generate reflexive context for a new query
                query = signal.payload.get("query", "")
                max_tokens = signal.payload.get("max_tokens", 2000)
                context = await self._crystal.generate_context(query, max_tokens)
                return CellSignal(
                    signal_type="crystal_response",
                    source_cell=self.config.cell_id,
                    target_cell=signal.source_cell,
                    payload={
                        "query_type": "context",
                        "context": context,
                        "has_context": bool(context),
                    },
                )
            
            else:
                return CellSignal(
                    signal_type="crystal_error",
                    source_cell=self.config.cell_id,
                    target_cell=signal.source_cell,
                    payload={"error": f"unknown_query_type: {query_type}"},
                )
                
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("💎 Crystal query failed: %s", e)
            return CellSignal(
                signal_type="crystal_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": str(e)},
            )
    
    async def _handle_get_crystal_stats(self, signal: CellSignal) -> CellSignal:
        """Get crystal statistics."""
        if not self._crystal:
            return CellSignal(
                signal_type="crystal_stats",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "crystal_not_available"},
            )
        
        try:
            stats = await self._crystal.get_stats()
            return CellSignal(
                signal_type="crystal_stats",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload=stats,
            )
        except (RuntimeError, OSError, ValueError) as e:
            return CellSignal(
                signal_type="crystal_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": str(e)},
            )
    
    async def _handle_rescore_crystal(self, signal: CellSignal) -> CellSignal:
        """
        Rescore all exchanges in the crystal using current scoring heuristics.
        
        Useful when:
        - Scoring algorithm has been updated
        - Manual quality adjustments need to be propagated
        - Evolutionary selection needs fresh fitness values
        """
        if not self._crystal or QualityScorer is None:
            return CellSignal(
                signal_type="crystal_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "crystal_or_scorer_not_available"},
            )
        
        try:
            # Get all processes
            all_processes = await self._crystal.get_all_processes()
            rescored_count = 0
            updated_processes = 0
            
            for process in all_processes:
                process_updated = False
                for exchange in process.exchanges:
                    if exchange.role == "response":
                        score, tier, _ = QualityScorer.score_exchange(exchange)
                        if exchange.quality_score != score or exchange.quality_tier != tier:
                            exchange.quality_score = score
                            exchange.quality_tier = tier
                            rescored_count += 1
                            process_updated = True
                
                if process_updated:
                    # Re-persist the updated process
                    await self._crystal.crystallize_process(process)
                    updated_processes += 1
            
            logger.info("💎 Rescored %d exchanges in %d processes", rescored_count, updated_processes)
            
            return CellSignal(
                signal_type="rescore_complete",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={
                    "rescored_exchanges": rescored_count,
                    "updated_processes": updated_processes,
                },
            )
            
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("💎 Rescoring failed: %s", e)
            return CellSignal(
                signal_type="crystal_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": str(e)},
            )
    
    async def _handle_get_quality_context(self, signal: CellSignal) -> CellSignal:
        """
        Generate quality-weighted reflexive context.
        
        Prioritizes HIGH and EXCEPTIONAL exchanges over LOW quality ones.
        This is the evolutionary selection pressure in action.
        """
        if not self._crystal:
            return CellSignal(
                signal_type="crystal_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "crystal_not_available"},
            )
        
        try:
            query = signal.payload.get("query", "")
            max_tokens = signal.payload.get("max_tokens", 3000)
            min_quality = signal.payload.get("min_quality", "MEDIUM")  # Filter threshold
            
            # Map string to enum
            quality_thresholds = {
                "UNKNOWN": ConversationQuality.UNKNOWN,
                "LOW": ConversationQuality.LOW,
                "MEDIUM": ConversationQuality.MEDIUM,
                "HIGH": ConversationQuality.HIGH,
                "EXCEPTIONAL": ConversationQuality.EXCEPTIONAL,
            }
            min_tier = quality_thresholds.get(min_quality.upper(), ConversationQuality.MEDIUM)
            
            # Quality tier order for comparison
            tier_order = [
                ConversationQuality.UNKNOWN,
                ConversationQuality.LOW,
                ConversationQuality.MEDIUM,
                ConversationQuality.HIGH,
                ConversationQuality.EXCEPTIONAL,
            ]
            min_tier_idx = tier_order.index(min_tier)
            
            # Get recent processes
            recent = await self._crystal.get_recent(limit=20)
            
            # Filter and weight by quality
            quality_weighted = []
            for process in recent:
                # Calculate average quality for this process
                scores = [
                    ex.quality_score for ex in process.exchanges
                    if ex.role == "response" and ex.quality_score > 0
                ]
                if not scores:
                    continue
                    
                avg_score = sum(scores) / len(scores)
                max_tier = max(
                    (ex.quality_tier for ex in process.exchanges if ex.role == "response"),
                    default=ConversationQuality.UNKNOWN
                )
                
                # Skip if below quality threshold
                if tier_order.index(max_tier) < min_tier_idx:
                    continue
                
                quality_weighted.append((process, avg_score, max_tier))
            
            # Sort by quality (highest first)
            quality_weighted.sort(key=lambda x: x[1], reverse=True)
            
            # Build context from highest quality first
            context_parts = []
            token_estimate = 0
            
            for process, score, tier in quality_weighted:
                process_context = f"\n[Prior Exchange - Quality: {tier.value} ({score:.2f})]\n"
                process_context += f"Query: {process.origin_query[:200]}...\n" if len(process.origin_query) > 200 else f"Query: {process.origin_query}\n"
                
                # Include best response
                best_response = max(
                    (ex for ex in process.exchanges if ex.role == "response"),
                    key=lambda x: x.quality_score,
                    default=None
                )
                if best_response:
                    response_preview = best_response.content[:500] + "..." if len(best_response.content) > 500 else best_response.content
                    process_context += f"Response: {response_preview}\n"
                
                # Rough token estimate (4 chars per token)
                context_tokens = len(process_context) // 4
                if token_estimate + context_tokens > max_tokens:
                    break
                    
                context_parts.append(process_context)
                token_estimate += context_tokens
            
            final_context = "\n".join(context_parts) if context_parts else ""
            
            return CellSignal(
                signal_type="quality_context_response",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={
                    "context": final_context,
                    "processes_included": len(context_parts),
                    "min_quality_filter": min_quality,
                    "token_estimate": token_estimate,
                },
            )
            
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("💎 Quality context generation failed: %s", e)
            return CellSignal(
                signal_type="crystal_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": str(e)},
            )
    
    def _select_agent_for_tier(self, tier: AgentTier) -> Optional[str]:
        """Select an appropriate agent for the given tier."""
        for agent_id, agent in self._agents.items():
            if isinstance(agent, OllamaAgent):
                if agent.tier == tier:
                    if self._agent_states.get(agent_id) == AgentState.READY:
                        return agent_id
            elif isinstance(agent, GeminiAgent):
                if agent.tier == tier:
                    if self._agent_states.get(agent_id) == AgentState.READY:
                        return agent_id
        
        # Fallback: find any agent at or below the tier
        tier_order = [AgentTier.LOCAL_FAST, AgentTier.LOCAL_REASONING, AgentTier.CLOUD_FAST, AgentTier.CLOUD_PRO]
        target_idx = tier_order.index(tier)
        
        for check_tier in tier_order[:target_idx + 1]:
            for agent_id, agent in self._agents.items():
                if hasattr(agent, 'tier') and agent.tier == check_tier:
                    if self._agent_states.get(agent_id) == AgentState.READY:
                        return agent_id
        
        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # CONVERSATION HANDLING
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _handle_start_conversation(self, signal: CellSignal) -> CellSignal:
        """Start a conversation with an embedded agent."""
        import uuid
        
        conv_id = uuid.uuid4().hex
        self._active_conversations[conv_id] = {
            "partner_cell": signal.source_cell,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "state": "active",
            "messages": [],
        }
        
        return CellSignal(
            signal_type="conversation_started",
            source_cell=self.config.cell_id,
            target_cell=signal.source_cell,
            payload={"conversation_id": conv_id},
        )
    
    async def _handle_conversation_message(self, signal: CellSignal) -> CellSignal:
        """Handle a message in an ongoing conversation."""
        conv_id = signal.payload.get("conversation_id")
        message = signal.payload.get("message", "")
        
        if conv_id not in self._active_conversations:
            return CellSignal(
                signal_type="conversation_error",
                source_cell=self.config.cell_id,
                target_cell=signal.source_cell,
                payload={"error": "conversation_not_found"},
            )
        
        # Route to query handler
        query_signal = CellSignal(
            id=signal.id,
            signal_type="query_agent",
            source_cell=signal.source_cell,
            payload={"query": message},
        )
        
        return await self._handle_query_agent(query_signal)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COHERENCE
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _update_coherence(self):
        """Update system-wide coherence score."""
        # Simple coherence based on agent availability and recent activity
        ready_ratio = (
            sum(1 for s in self._agent_states.values() if s == AgentState.READY)
            / max(len(self._agents), 1)
        )
        
        active_conclaves = sum(
            1 for c in self._conclaves.values()
            if c.state in [ConclaveState.FORMING, ConclaveState.DELIBERATING]
        )
        
        # Coherence decreases with violations, increases with activity
        self._coherence_score = min(1.0, ready_ratio * 0.7 + 0.3)
    
    def _trim_thoughts(self):
        """Trim thought history to max size."""
        if len(self._thoughts) > self._max_thoughts:
            self._thoughts = self._thoughts[-self._max_thoughts:]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PROMETHEUS METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _collect_cell_metrics(self) -> Dict[str, float]:
        """Collect Thinker-specific metrics for Prometheus."""
        metrics = {
            # Agent metrics
            "agents_embedded": float(len(self._agents)),
            "agents_ready": float(
                sum(1 for s in self._agent_states.values() if s == AgentState.READY)
            ),
            "coherence_score": float(self._coherence_score),
            "coherence_violations": float(self._coherence_violations),
            
            # Conversation metrics
            "active_conversations": float(len(self._active_conversations)),
            "active_conclaves": float(len(self._conclaves)),
            "thought_count": float(len(self._thoughts)),
            
            # Fractal process metrics
            "fractal_processes": float(len(self._fractal_processes)),
        }
        
        # Crystal metrics if available
        if self._crystal:
            try:
                stats = await self._crystal.get_stats()
                metrics.update({
                    "crystal_total_processes": float(stats.get("total_processes", 0)),
                    "crystal_total_exchanges": float(stats.get("total_exchanges", 0)),
                    "crystal_average_quality": float(stats.get("average_quality", 0.0)),
                    "crystal_peak_quality": float(stats.get("peak_quality", 0.0)),
                    "crystal_tier_exceptional": float(stats.get("tier_distribution", {}).get("exceptional", 0)),
                    "crystal_tier_high": float(stats.get("tier_distribution", {}).get("high", 0)),
                    "crystal_tier_medium": float(stats.get("tier_distribution", {}).get("medium", 0)),
                    "crystal_tier_low": float(stats.get("tier_distribution", {}).get("low", 0)),
                    "crystal_tier_minimal": float(stats.get("tier_distribution", {}).get("minimal", 0)),
                })
            except (RuntimeError, OSError, ValueError):
                pass  # Crystal stats unavailable
        
        return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Run Thinker cell standalone."""
    config = CellConfig.from_env()
    config.cell_type = CellType.THINKER
    
    cell = ThinkerCell(config)
    
    try:
        await cell.run_forever()
    except KeyboardInterrupt:
        await cell.shutdown()


if __name__ == "__main__":
    print("🧠 Starting Thinker Cell...")
    asyncio.run(main())

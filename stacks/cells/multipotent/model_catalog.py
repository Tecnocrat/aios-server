"""
AIOS Multi-Agent Model Catalog
==============================

Organized by tier and use case for predictable, intelligent orchestration.
Focus: Local Ollama + GitHub Models API (Microsoft subscription)

DESIGN PHILOSOPHY:
- Smaller models = more predictable outputs for structured tasks
- Older models = well-understood behavior patterns
- Tiered escalation = cost/quality optimization
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict


class ModelProvider(Enum):
    """Where the model runs."""
    OLLAMA = "ollama"           # Local, free
    GITHUB = "github"           # GitHub Models API (subscription)
    AZURE = "azure"             # Azure OpenAI (if needed)


class TaskType(Enum):
    """What the model is good at."""
    GENERAL = "general"
    CODE = "code"
    MATH = "math"
    REASONING = "reasoning"
    EXTRACTION = "extraction"   # Structured output extraction
    CHAT = "chat"
    EMBEDDING = "embedding"
    VISION = "vision"


class Predictability(Enum):
    """How deterministic the model's outputs are."""
    HIGH = "high"       # Small, old, fine-tuned - very consistent
    MEDIUM = "medium"   # Moderate variation
    LOW = "low"         # Large, creative - more variable


@dataclass
class ModelSpec:
    """Specification for a model."""
    name: str
    provider: ModelProvider
    size: str                           # e.g., "1b", "7b", "270m"
    tier: str                           # e.g., "LOCAL_FAST", "LOCAL_REASONING"
    tasks: List[TaskType]
    predictability: Predictability
    context_length: int = 4096
    description: str = ""
    pull_command: str = ""              # For Ollama
    cost_per_1m_tokens: float = 0.0     # 0 = free
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 0: MICRO (< 500M params) - Highest Predictability
# ═══════════════════════════════════════════════════════════════════════════════
# Best for: Structured extraction, simple classification, deterministic tasks

TIER_0_MICRO = [
    ModelSpec(
        name="gemma3:270m",
        provider=ModelProvider.OLLAMA,
        size="270m",
        tier="MICRO",
        tasks=[TaskType.GENERAL, TaskType.EXTRACTION],
        predictability=Predictability.HIGH,
        context_length=8192,
        description="Tiny Gemma for simple tasks",
        pull_command="ollama pull gemma3:270m",
        notes="Great for structured output, classification"
    ),
    ModelSpec(
        name="smollm2:135m",
        provider=ModelProvider.OLLAMA,
        size="135m",
        tier="MICRO",
        tasks=[TaskType.GENERAL],
        predictability=Predictability.HIGH,
        context_length=2048,
        description="Smallest viable LLM",
        pull_command="ollama pull smollm2:135m",
        notes="Ultra-fast, minimal memory"
    ),
    ModelSpec(
        name="smollm2:360m",
        provider=ModelProvider.OLLAMA,
        size="360m",
        tier="MICRO",
        tasks=[TaskType.GENERAL],
        predictability=Predictability.HIGH,
        context_length=2048,
        pull_command="ollama pull smollm2:360m",
        notes="Slightly more capable than 135m"
    ),
    ModelSpec(
        name="functiongemma:270m",
        provider=ModelProvider.OLLAMA,
        size="270m",
        tier="MICRO",
        tasks=[TaskType.EXTRACTION],
        predictability=Predictability.HIGH,
        context_length=8192,
        description="Function calling specialist",
        pull_command="ollama pull functiongemma",
        notes="EXCELLENT for tool/function extraction"
    ),
    ModelSpec(
        name="qwen:0.5b",
        provider=ModelProvider.OLLAMA,
        size="0.5b",
        tier="MICRO",
        tasks=[TaskType.GENERAL],
        predictability=Predictability.HIGH,
        context_length=32768,
        pull_command="ollama pull qwen:0.5b",
        notes="Old but reliable, long context"
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1: LOCAL_FAST (1-2B params) - High Predictability
# ═══════════════════════════════════════════════════════════════════════════════
# Best for: Quick iteration, simple Q&A, summarization

TIER_1_LOCAL_FAST = [
    ModelSpec(
        name="gemma3:1b",
        provider=ModelProvider.OLLAMA,
        size="1b",
        tier="LOCAL_FAST",
        tasks=[TaskType.GENERAL, TaskType.CHAT],
        predictability=Predictability.HIGH,
        context_length=8192,
        description="Fast, modern, reliable",
        pull_command="ollama pull gemma3:1b",
        notes="CURRENTLY INSTALLED - Good balance"
    ),
    ModelSpec(
        name="tinyllama:1.1b",
        provider=ModelProvider.OLLAMA,
        size="1.1b",
        tier="LOCAL_FAST",
        tasks=[TaskType.GENERAL, TaskType.CHAT],
        predictability=Predictability.HIGH,
        context_length=2048,
        description="Classic small model",
        pull_command="ollama pull tinyllama",
        notes="OLD BUT PREDICTABLE - trained on 3T tokens"
    ),
    ModelSpec(
        name="tinydolphin:1.1b",
        provider=ModelProvider.OLLAMA,
        size="1.1b",
        tier="LOCAL_FAST",
        tasks=[TaskType.CHAT],
        predictability=Predictability.HIGH,
        context_length=2048,
        description="Uncensored chat variant",
        pull_command="ollama pull tinydolphin",
        notes="Good for unrestricted responses"
    ),
    ModelSpec(
        name="qwen2.5:1.5b",
        provider=ModelProvider.OLLAMA,
        size="1.5b",
        tier="LOCAL_FAST",
        tasks=[TaskType.GENERAL, TaskType.CODE],
        predictability=Predictability.HIGH,
        context_length=131072,
        pull_command="ollama pull qwen2.5:1.5b",
        notes="128K context! Good for long docs"
    ),
    ModelSpec(
        name="llama3.2:1b",
        provider=ModelProvider.OLLAMA,
        size="1b",
        tier="LOCAL_FAST",
        tasks=[TaskType.GENERAL],
        predictability=Predictability.HIGH,
        context_length=131072,
        pull_command="ollama pull llama3.2:1b",
        notes="Meta's small model, 128K context"
    ),
    ModelSpec(
        name="deepseek-coder:1.3b",
        provider=ModelProvider.OLLAMA,
        size="1.3b",
        tier="LOCAL_FAST",
        tasks=[TaskType.CODE],
        predictability=Predictability.HIGH,
        context_length=16384,
        pull_command="ollama pull deepseek-coder:1.3b",
        notes="EXCELLENT for simple code tasks"
    ),
    ModelSpec(
        name="phi:2.7b",
        provider=ModelProvider.OLLAMA,
        size="2.7b",
        tier="LOCAL_FAST",
        tasks=[TaskType.REASONING, TaskType.MATH],
        predictability=Predictability.HIGH,
        context_length=2048,
        description="Microsoft's reasoning model",
        pull_command="ollama pull phi",
        notes="OLD - Great for logic/math, very predictable"
    ),
    ModelSpec(
        name="stablelm2:1.6b",
        provider=ModelProvider.OLLAMA,
        size="1.6b",
        tier="LOCAL_FAST",
        tasks=[TaskType.GENERAL],
        predictability=Predictability.HIGH,
        context_length=4096,
        pull_command="ollama pull stablelm2:1.6b",
        notes="Multilingual support"
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2: LOCAL_REASONING (3-8B params) - Medium Predictability
# ═══════════════════════════════════════════════════════════════════════════════
# Best for: Complex reasoning, code generation, analysis

TIER_2_LOCAL_REASONING = [
    ModelSpec(
        name="mistral:7b",
        provider=ModelProvider.OLLAMA,
        size="7b",
        tier="LOCAL_REASONING",
        tasks=[TaskType.GENERAL, TaskType.REASONING],
        predictability=Predictability.MEDIUM,
        context_length=32768,
        description="Excellent all-rounder",
        pull_command="ollama pull mistral:7b",
        notes="CURRENTLY INSTALLED - Good reasoning"
    ),
    ModelSpec(
        name="gemma3:4b",
        provider=ModelProvider.OLLAMA,
        size="4b",
        tier="LOCAL_REASONING",
        tasks=[TaskType.GENERAL, TaskType.CODE],
        predictability=Predictability.MEDIUM,
        context_length=8192,
        pull_command="ollama pull gemma3:4b",
        notes="Good balance of speed/quality"
    ),
    ModelSpec(
        name="phi3:3.8b",
        provider=ModelProvider.OLLAMA,
        size="3.8b",
        tier="LOCAL_REASONING",
        tasks=[TaskType.REASONING, TaskType.MATH],
        predictability=Predictability.MEDIUM,
        context_length=4096,
        description="Microsoft Phi-3 Mini",
        pull_command="ollama pull phi3",
        notes="Strong reasoning, compact"
    ),
    ModelSpec(
        name="llama3.2:3b",
        provider=ModelProvider.OLLAMA,
        size="3b",
        tier="LOCAL_REASONING",
        tasks=[TaskType.GENERAL],
        predictability=Predictability.MEDIUM,
        context_length=131072,
        pull_command="ollama pull llama3.2:3b",
        notes="Meta's latest small, 128K context"
    ),
    ModelSpec(
        name="codellama:7b",
        provider=ModelProvider.OLLAMA,
        size="7b",
        tier="LOCAL_REASONING",
        tasks=[TaskType.CODE],
        predictability=Predictability.MEDIUM,
        context_length=16384,
        description="Code specialist",
        pull_command="ollama pull codellama:7b",
        notes="EXCELLENT for code generation"
    ),
    ModelSpec(
        name="deepseek-coder:6.7b",
        provider=ModelProvider.OLLAMA,
        size="6.7b",
        tier="LOCAL_REASONING",
        tasks=[TaskType.CODE],
        predictability=Predictability.MEDIUM,
        context_length=16384,
        pull_command="ollama pull deepseek-coder:6.7b",
        notes="Strong coder, newer than codellama"
    ),
    ModelSpec(
        name="llama3:8b",
        provider=ModelProvider.OLLAMA,
        size="8b",
        tier="LOCAL_REASONING",
        tasks=[TaskType.GENERAL, TaskType.REASONING],
        predictability=Predictability.MEDIUM,
        context_length=8192,
        pull_command="ollama pull llama3:8b",
        notes="Solid, well-tested"
    ),
    ModelSpec(
        name="llama2:7b",
        provider=ModelProvider.OLLAMA,
        size="7b",
        tier="LOCAL_REASONING",
        tasks=[TaskType.GENERAL],
        predictability=Predictability.HIGH,  # OLD = more predictable
        context_length=4096,
        pull_command="ollama pull llama2:7b",
        notes="OLD BUT VERY PREDICTABLE"
    ),
    ModelSpec(
        name="orca-mini:3b",
        provider=ModelProvider.OLLAMA,
        size="3b",
        tier="LOCAL_REASONING",
        tasks=[TaskType.GENERAL],
        predictability=Predictability.HIGH,
        context_length=2048,
        pull_command="ollama pull orca-mini:3b",
        notes="OLD - Good for simple reasoning tasks"
    ),
    ModelSpec(
        name="neural-chat:7b",
        provider=ModelProvider.OLLAMA,
        size="7b",
        tier="LOCAL_REASONING",
        tasks=[TaskType.CHAT],
        predictability=Predictability.HIGH,
        context_length=8192,
        pull_command="ollama pull neural-chat",
        notes="OLD Intel model - Very stable chat"
    ),
    ModelSpec(
        name="vicuna:7b",
        provider=ModelProvider.OLLAMA,
        size="7b",
        tier="LOCAL_REASONING",
        tasks=[TaskType.CHAT],
        predictability=Predictability.HIGH,
        context_length=2048,
        pull_command="ollama pull vicuna:7b",
        notes="CLASSIC - Very predictable chat"
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3: GITHUB_MODELS (Microsoft/GitHub subscription)
# ═══════════════════════════════════════════════════════════════════════════════
# Best for: When local isn't enough, need cloud quality

TIER_3_GITHUB = [
    ModelSpec(
        name="gpt-4o-mini",
        provider=ModelProvider.GITHUB,
        size="small",
        tier="GITHUB_FAST",
        tasks=[TaskType.GENERAL, TaskType.CODE, TaskType.REASONING],
        predictability=Predictability.MEDIUM,
        context_length=128000,
        cost_per_1m_tokens=0.15,  # input
        notes="GitHub Models - Fast, cheap, good"
    ),
    ModelSpec(
        name="gpt-4o",
        provider=ModelProvider.GITHUB,
        size="large",
        tier="GITHUB_PRO",
        tasks=[TaskType.GENERAL, TaskType.CODE, TaskType.REASONING, TaskType.VISION],
        predictability=Predictability.MEDIUM,
        context_length=128000,
        cost_per_1m_tokens=5.0,  # input
        notes="GitHub Models - Top tier quality"
    ),
    ModelSpec(
        name="Phi-3.5-mini-instruct",
        provider=ModelProvider.GITHUB,
        size="3.5b",
        tier="GITHUB_FAST",
        tasks=[TaskType.GENERAL, TaskType.REASONING],
        predictability=Predictability.MEDIUM,
        context_length=128000,
        cost_per_1m_tokens=0.0,  # Free tier?
        notes="Microsoft model on GitHub - Free?"
    ),
    ModelSpec(
        name="Mistral-small",
        provider=ModelProvider.GITHUB,
        size="22b",
        tier="GITHUB_REASONING",
        tasks=[TaskType.GENERAL, TaskType.REASONING],
        predictability=Predictability.MEDIUM,
        context_length=32000,
        notes="Mistral on GitHub Models"
    ),
    ModelSpec(
        name="Llama-3.2-11B-Vision-Instruct",
        provider=ModelProvider.GITHUB,
        size="11b",
        tier="GITHUB_VISION",
        tasks=[TaskType.VISION, TaskType.GENERAL],
        predictability=Predictability.MEDIUM,
        context_length=128000,
        notes="Vision capability on GitHub"
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING MODELS
# ═══════════════════════════════════════════════════════════════════════════════

EMBEDDING_MODELS = [
    ModelSpec(
        name="nomic-embed-text",
        provider=ModelProvider.OLLAMA,
        size="137m",
        tier="EMBEDDING",
        tasks=[TaskType.EMBEDDING],
        predictability=Predictability.HIGH,
        context_length=8192,
        pull_command="ollama pull nomic-embed-text",
        notes="Best local embedding model"
    ),
    ModelSpec(
        name="all-minilm:22m",
        provider=ModelProvider.OLLAMA,
        size="22m",
        tier="EMBEDDING",
        tasks=[TaskType.EMBEDDING],
        predictability=Predictability.HIGH,
        context_length=512,
        pull_command="ollama pull all-minilm:22m",
        notes="Tiny, fast embedding"
    ),
    ModelSpec(
        name="mxbai-embed-large",
        provider=ModelProvider.OLLAMA,
        size="335m",
        tier="EMBEDDING",
        tasks=[TaskType.EMBEDDING],
        predictability=Predictability.HIGH,
        context_length=512,
        pull_command="ollama pull mxbai-embed-large",
        notes="High quality embeddings"
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# RECOMMENDED MINIMAL SET (to pull now)
# ═══════════════════════════════════════════════════════════════════════════════

RECOMMENDED_MODELS = """
# Already installed:
✓ gemma3:1b      (815 MB)  - LOCAL_FAST
✓ mistral:7b     (4.4 GB)  - LOCAL_REASONING

# Recommended to pull for complete tier coverage:

## MICRO tier (highest predictability):
ollama pull smollm2:135m        # 98 MB - Ultra tiny
ollama pull functiongemma       # 183 MB - Function extraction

## LOCAL_FAST additions:
ollama pull tinyllama           # 637 MB - Classic, predictable
ollama pull phi                 # 1.6 GB - MS reasoning (old)
ollama pull deepseek-coder:1.3b # 776 MB - Code tasks

## LOCAL_REASONING additions:
ollama pull codellama:7b        # 3.8 GB - Code specialist
ollama pull llama2:7b           # 3.8 GB - OLD, very predictable
ollama pull orca-mini:3b        # 1.9 GB - Simple reasoning

## Embeddings:
ollama pull nomic-embed-text    # 274 MB - Best embeddings

# Total additional: ~13 GB
"""


def get_recommended_pulls() -> List[str]:
    """Get list of recommended ollama pull commands."""
    return [
        "ollama pull smollm2:135m",
        "ollama pull functiongemma", 
        "ollama pull tinyllama",
        "ollama pull phi",
        "ollama pull deepseek-coder:1.3b",
        "ollama pull codellama:7b",
        "ollama pull llama2:7b",
        "ollama pull orca-mini:3b",
        "ollama pull nomic-embed-text",
    ]


def get_model_for_task(task: TaskType, prefer_predictable: bool = True) -> List[ModelSpec]:
    """Get models suitable for a task, sorted by predictability."""
    all_models = TIER_0_MICRO + TIER_1_LOCAL_FAST + TIER_2_LOCAL_REASONING
    suitable = [m for m in all_models if task in m.tasks]
    
    if prefer_predictable:
        # Sort by predictability (HIGH first)
        pred_order = {Predictability.HIGH: 0, Predictability.MEDIUM: 1, Predictability.LOW: 2}
        suitable.sort(key=lambda m: pred_order[m.predictability])
    
    return suitable


if __name__ == "__main__":
    print("=" * 70)
    print("AIOS MODEL CATALOG")
    print("=" * 70)
    print(RECOMMENDED_MODELS)
    
    print("\n" + "=" * 70)
    print("MODELS BY PREDICTABILITY")
    print("=" * 70)
    
    all_models = TIER_0_MICRO + TIER_1_LOCAL_FAST + TIER_2_LOCAL_REASONING
    
    for pred in [Predictability.HIGH, Predictability.MEDIUM]:
        print(f"\n{pred.value.upper()} PREDICTABILITY:")
        for m in all_models:
            if m.predictability == pred:
                tasks_str = ", ".join(t.value for t in m.tasks)
                print(f"  {m.name:25} | {m.size:6} | {tasks_str}")

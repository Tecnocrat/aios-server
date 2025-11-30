#!/usr/bin/env python3
"""
AIOS Shared Dendritic Utilities
Centralized framework detection and fallback utilities for AIOS components
"""

import importlib.util
import logging
from typing import Any, Dict, Type

logger = logging.getLogger(__name__)

# AINLP.dendritic: Top-level optional import with graceful fallback
try:
    from pydantic import BaseModel as PydanticBaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PydanticBaseModel = None  # type: ignore
    PYDANTIC_AVAILABLE = False


class DendriticFrameworkDetector:
    """AINLP.dendritic growth: Centralized framework availability detection"""

    def __init__(self) -> None:
        self._cache: Dict[str, bool] = {}

    def is_available(self, framework_name: str) -> bool:
        """Check if a framework is available using enhanced dendritic logic"""
        if framework_name in self._cache:
            return self._cache[framework_name]

        try:
            spec = importlib.util.find_spec(framework_name)
            available = spec is not None
            self._cache[framework_name] = available
            return available
        except (ModuleNotFoundError, ValueError, ImportError) as exc:
            logger.debug("Framework %s unavailable: %s", framework_name, exc)
            self._cache[framework_name] = False
            return False

    def get_available_frameworks(self, frameworks: list) -> Dict[str, bool]:
        """Check availability of multiple frameworks"""
        return {fw: self.is_available(fw) for fw in frameworks}


def create_fallback_app(title: str = "AIOS Fallback App") -> Dict[str, Any]:
    """AINLP.dendritic: Create fallback app when FastAPI unavailable"""
    logger.warning("AINLP.dendritic: Using pure Python fallback app")
    return {
        "type": "fallback",
        "framework": "none",
        "title": title,
        "routes": []
    }


class FallbackBaseModel:
    """AINLP.dendritic: Fallback BaseModel when pydantic unavailable"""

    def __init__(self, **data: Any) -> None:
        for key, value in data.items():
            setattr(self, key, value)


def get_base_model() -> Type:
    """AINLP.dendritic: Get BaseModel class with fallback"""
    if PYDANTIC_AVAILABLE and PydanticBaseModel is not None:
        return PydanticBaseModel
    logger.warning("AINLP.dendritic: Pydantic unavailable, using fallback")
    return FallbackBaseModel

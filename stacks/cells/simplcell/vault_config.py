#!/usr/bin/env python3
"""
AIOS Vault-Aware Configuration Provider

Implements the AI Agent Vault Protocol for SimplCells.
Provides dynamic configuration discovery from Vault with graceful fallback.

Phase 31.5.17: Vault Integration for Cellular Architecture
AINLP.cellular[VAULT_CONFIG] Semantic pointer system for cell configuration

Core Principle: "Never hardcode. Always query Vault first."
Fallback: ENV variables when Vault unavailable (development mode)

Usage:
    from vault_config import VaultConfig
    
    config = VaultConfig()
    ollama_host = config.get("ollama_host", default="http://localhost:11434")
    oracle_url = config.get("oracle_url")  # Returns None if not found
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("VaultConfig")


@dataclass
class VaultPaths:
    """Semantic pointer paths for AIOS configuration in Vault."""
    # System-level configuration
    SYSTEM_PATHS = "aios-secrets/system/paths"
    SYSTEM_VAULT = "aios-secrets/system/vault"
    SYSTEM_ENDPOINTS = "aios-secrets/system/endpoints"
    SYSTEM_BACKUP = "aios-secrets/system/backup"
    
    # Service credentials
    GRAFANA = "aios-secrets/grafana"
    PROMETHEUS = "aios-secrets/prometheus"
    TRAEFIK = "aios-secrets/traefik"
    
    # Cell-specific configuration (NEW - Phase 31.5.17)
    CELLS_BASE = "aios-secrets/cells"
    CELLS_ENDPOINTS = "aios-secrets/cells/endpoints"
    CELLS_ORACLE = "aios-secrets/cells/oracle"
    CELLS_GENOME = "aios-secrets/cells/genome"


class VaultConfig:
    """
    Vault-aware configuration provider for AIOS cells.
    
    Implements the AI Agent Vault Protocol:
    1. Bootstrap from single hardcoded path (vault token file)
    2. Discover all configuration via Vault queries
    3. Graceful fallback to ENV vars when Vault unavailable
    
    This allows cells to be deployed:
    - In production: Full Vault integration for secrets management
    - In development: ENV-var based for quick iteration
    """
    
    # Bootstrap path - THE ONLY hardcoded value
    VAULT_TOKEN_PATH = "/config/vault-root-token.txt"
    VAULT_CONTAINER = "aios-vault"
    
    def __init__(self, cell_id: str = None):
        """
        Initialize Vault configuration provider.
        
        Args:
            cell_id: Optional cell identifier for cell-specific config
        """
        self.cell_id = cell_id or os.getenv("AIOS_CELL_ID", "unknown")
        self._vault_available = None
        self._token = None
        self._cache: Dict[str, Any] = {}
        
        # Check Vault availability on init
        self._check_vault_availability()
    
    def _check_vault_availability(self) -> bool:
        """Check if Vault is accessible."""
        if self._vault_available is not None:
            return self._vault_available
        
        # Try to get token
        token_paths = [
            self.VAULT_TOKEN_PATH,
            "/app/config/vault-root-token.txt",
            "./config/vault-root-token.txt",
            os.getenv("VAULT_TOKEN_FILE", ""),
        ]
        
        for path in token_paths:
            if path and Path(path).exists():
                try:
                    self._token = Path(path).read_text().strip()
                    break
                except Exception:
                    continue
        
        # Also check ENV
        if not self._token:
            self._token = os.getenv("VAULT_TOKEN")
        
        if not self._token:
            logger.info("🔐 Vault token not found - using ENV fallback mode")
            self._vault_available = False
            return False
        
        # Test Vault connectivity
        try:
            result = subprocess.run(
                ["docker", "exec", "-e", f"VAULT_TOKEN={self._token}",
                 self.VAULT_CONTAINER, "vault", "status"],
                capture_output=True, timeout=5
            )
            self._vault_available = result.returncode == 0
            
            if self._vault_available:
                logger.info("🔐 Vault connected - using semantic pointer mode")
            else:
                logger.warning("🔐 Vault unreachable - using ENV fallback mode")
                
        except Exception as e:
            logger.warning(f"🔐 Vault check failed ({e}) - using ENV fallback mode")
            self._vault_available = False
        
        return self._vault_available
    
    def _query_vault(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Query Vault for configuration at given path.
        
        Args:
            path: Vault secret path (e.g., "aios-secrets/system/endpoints")
            
        Returns:
            Dictionary of key-value pairs, or None if not found
        """
        if not self._vault_available or not self._token:
            return None
        
        # Check cache first
        if path in self._cache:
            return self._cache[path]
        
        try:
            result = subprocess.run(
                ["docker", "exec", "-e", f"VAULT_TOKEN={self._token}",
                 self.VAULT_CONTAINER, "vault", "kv", "get",
                 "-format=json", path],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                secrets = data.get("data", {}).get("data", {})
                self._cache[path] = secrets
                logger.debug(f"🔐 Vault query successful: {path}")
                return secrets
            else:
                logger.debug(f"🔐 Vault path not found: {path}")
                return None
                
        except Exception as e:
            logger.warning(f"🔐 Vault query failed for {path}: {e}")
            return None
    
    def get(self, key: str, default: Any = None, vault_path: str = None) -> Any:
        """
        Get configuration value using Vault Protocol.
        
        Resolution order:
        1. Cell-specific Vault path (aios-secrets/cells/{cell_id}/{key})
        2. Specified vault_path if provided
        3. Common Vault paths (endpoints, paths, etc.)
        4. Environment variable (AIOS_{KEY} or {KEY})
        5. Default value
        
        Args:
            key: Configuration key name
            default: Default value if not found
            vault_path: Optional specific Vault path to query
            
        Returns:
            Configuration value or default
        """
        # Strategy 1: Try cell-specific Vault path
        if self._vault_available:
            cell_secrets = self._query_vault(f"{VaultPaths.CELLS_BASE}/{self.cell_id}")
            if cell_secrets and key in cell_secrets:
                return cell_secrets[key]
        
        # Strategy 2: Try specified vault_path
        if vault_path and self._vault_available:
            secrets = self._query_vault(vault_path)
            if secrets and key in secrets:
                return secrets[key]
        
        # Strategy 3: Try common Vault paths
        if self._vault_available:
            common_paths = [
                VaultPaths.CELLS_ENDPOINTS,
                VaultPaths.SYSTEM_ENDPOINTS,
                VaultPaths.SYSTEM_PATHS,
            ]
            for path in common_paths:
                secrets = self._query_vault(path)
                if secrets and key in secrets:
                    return secrets[key]
        
        # Strategy 4: Try environment variables
        env_keys = [
            f"AIOS_{key.upper()}",
            key.upper(),
            key,
        ]
        for env_key in env_keys:
            value = os.getenv(env_key)
            if value is not None:
                return value
        
        # Strategy 5: Return default
        return default
    
    def get_endpoints(self) -> Dict[str, str]:
        """Get all service endpoints from Vault or ENV."""
        if self._vault_available:
            endpoints = self._query_vault(VaultPaths.SYSTEM_ENDPOINTS)
            if endpoints:
                return endpoints
        
        # Fallback to ENV-based discovery
        return {
            "ollama": os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"),
            "nous": os.getenv("NOUS_URL", "http://nous:8000"),
            "discovery": os.getenv("AIOS_DISCOVERY_ADDR", "aios-discovery:8001"),
            "vault": os.getenv("VAULT_ADDR", "http://aios-vault:8200"),
            "grafana": os.getenv("GRAFANA_URL", "http://grafana:3000"),
        }
    
    def get_oracle_config(self) -> Dict[str, Any]:
        """Get oracle (Nous) configuration for SimplCells."""
        if self._vault_available:
            oracle = self._query_vault(VaultPaths.CELLS_ORACLE)
            if oracle:
                return oracle
        
        # Fallback to ENV
        return {
            "url": os.getenv("ORACLE_URL", os.getenv("NOUS_URL", "")),
            "query_chance": float(os.getenv("ORACLE_QUERY_CHANCE", "0.1")),
            "enabled": os.getenv("ORACLE_ENABLED", "true").lower() == "true",
        }
    
    def get_cell_genome(self) -> Dict[str, Any]:
        """Get cell genome parameters from Vault or ENV."""
        if self._vault_available:
            # Try cell-specific genome first
            genome = self._query_vault(f"{VaultPaths.CELLS_BASE}/{self.cell_id}/genome")
            if genome:
                return genome
            
            # Try default genome
            genome = self._query_vault(VaultPaths.CELLS_GENOME)
            if genome:
                return genome
        
        # Fallback to ENV-based genome
        return {
            "temperature": float(os.getenv("CELL_TEMPERATURE", "0.7")),
            "model": os.getenv("CELL_MODEL", "llama3.2:3b"),
            "response_style": os.getenv("CELL_RESPONSE_STYLE", "concise"),
            "heartbeat_seconds": int(os.getenv("CELL_HEARTBEAT_SECONDS", "300")),
        }
    
    @property
    def is_vault_mode(self) -> bool:
        """Check if running with Vault integration."""
        return self._vault_available
    
    def status(self) -> Dict[str, Any]:
        """Return configuration provider status."""
        return {
            "cell_id": self.cell_id,
            "vault_available": self._vault_available,
            "mode": "vault" if self._vault_available else "env_fallback",
            "cached_paths": list(self._cache.keys()),
        }


# Convenience singleton for module-level access
_config_instance: Optional[VaultConfig] = None


def get_config(cell_id: str = None) -> VaultConfig:
    """Get or create the singleton VaultConfig instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = VaultConfig(cell_id)
    return _config_instance


# Quick access functions
def vault_get(key: str, default: Any = None) -> Any:
    """Quick access to configuration values."""
    return get_config().get(key, default)


def vault_endpoints() -> Dict[str, str]:
    """Quick access to service endpoints."""
    return get_config().get_endpoints()


if __name__ == "__main__":
    # Test Vault configuration
    logging.basicConfig(level=logging.DEBUG)
    config = VaultConfig(cell_id="test-cell")
    
    print("\n🔐 AIOS Vault Configuration Status")
    print("=" * 50)
    print(f"Mode: {config.status()['mode']}")
    print(f"Vault Available: {config.is_vault_mode}")
    
    print("\n📍 Endpoints:")
    for key, value in config.get_endpoints().items():
        print(f"  {key}: {value}")
    
    print("\n🧬 Cell Genome:")
    for key, value in config.get_cell_genome().items():
        print(f"  {key}: {value}")
    
    print("\n🔮 Oracle Config:")
    for key, value in config.get_oracle_config().items():
        print(f"  {key}: {value}")

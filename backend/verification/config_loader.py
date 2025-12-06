"""
Noise Configuration Loader — Load and Validate Noise Configs

This module loads and validates noise configuration from YAML files.
Provides type-safe access to noise parameters for slices and tiers.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from backend.verification.noise_sampler import (
    NoiseConfig,
    TimeoutDistributionConfig,
    TimeoutDistribution,
)


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "verifier_noise_phase2.yaml"


class NoiseConfigLoader:
    """Loader for noise configuration from YAML files."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize config loader.
        
        Args:
            config_path: Path to noise config YAML file (default: verifier_noise_phase2.yaml)
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._config_data: Optional[Dict[str, Any]] = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self._config_data is not None:
            return self._config_data
        
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Noise config not found at {self.config_path}. "
                "Please create config/verifier_noise_phase2.yaml"
            )
        
        with open(self.config_path, "r") as f:
            self._config_data = yaml.safe_load(f)
        
        return self._config_data
    
    def get_global_config(self) -> Dict[str, Any]:
        """Get global noise configuration."""
        config = self._load_config()
        return config.get("global", {})
    
    def get_tier_config(self, tier: str) -> Dict[str, Any]:
        """Get configuration for a specific tier.
        
        Args:
            tier: Tier name (fast_noisy, balanced, slow_precise)
        
        Returns:
            Tier configuration dict
        
        Raises:
            KeyError: If tier not found in config
        """
        config = self._load_config()
        tiers = config.get("tiers", {})
        
        if tier not in tiers:
            raise KeyError(f"Tier '{tier}' not found in noise config")
        
        return tiers[tier]
    
    def get_slice_config(self, slice_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific slice.
        
        Args:
            slice_name: Slice name (e.g., "arithmetic_simple")
        
        Returns:
            Slice configuration dict, or None if not found
        """
        config = self._load_config()
        slices = config.get("slices", {})
        return slices.get(slice_name)
    
    def load_noise_config_for_tier(
        self,
        tier: str,
        slice_name: Optional[str] = None,
    ) -> NoiseConfig:
        """Load NoiseConfig for a tier with optional slice overrides.
        
        Args:
            tier: Tier name (fast_noisy, balanced, slow_precise)
            slice_name: Optional slice name for overrides
        
        Returns:
            NoiseConfig instance
        """
        # Load base tier config
        tier_config = self.get_tier_config(tier)
        
        # Apply slice overrides if present
        if slice_name is not None:
            slice_config = self.get_slice_config(slice_name)
            if slice_config is not None:
                tier_overrides = slice_config.get("tier_overrides", {}).get(tier, {})
                tier_config = {**tier_config, **tier_overrides}
        
        # Parse timeout distribution
        timeout_dist_data = tier_config.get("timeout_distribution", {})
        timeout_dist = self._parse_timeout_distribution(timeout_dist_data)
        
        # Build NoiseConfig
        return NoiseConfig(
            noise_enabled=tier_config.get("noise_enabled", True),
            timeout_rate=tier_config.get("timeout_rate", 0.0),
            spurious_fail_rate=tier_config.get("spurious_fail_rate", 0.0),
            spurious_pass_rate=tier_config.get("spurious_pass_rate", 0.0),
            timeout_distribution=timeout_dist,
        )
    
    def _parse_timeout_distribution(
        self,
        dist_data: Dict[str, Any],
    ) -> TimeoutDistributionConfig:
        """Parse timeout distribution config from dict.
        
        Args:
            dist_data: Distribution config dict from YAML
        
        Returns:
            TimeoutDistributionConfig instance
        """
        dist_type_str = dist_data.get("type", "uniform")
        
        try:
            dist_type = TimeoutDistribution(dist_type_str)
        except ValueError:
            raise ValueError(f"Unknown timeout distribution type: {dist_type_str}")
        
        if dist_type == TimeoutDistribution.UNIFORM:
            return TimeoutDistributionConfig.uniform(
                min_ms=dist_data.get("min_ms", 500),
                max_ms=dist_data.get("max_ms", 1500),
            )
        elif dist_type == TimeoutDistribution.EXPONENTIAL:
            return TimeoutDistributionConfig.exponential(
                mean_ms=dist_data.get("mean_ms", 1000),
            )
        elif dist_type == TimeoutDistribution.FIXED:
            return TimeoutDistributionConfig.fixed(
                fixed_ms=dist_data.get("fixed_ms", 1000),
            )
        else:
            raise ValueError(f"Unsupported timeout distribution type: {dist_type}")
    
    def get_escalation_policy(self) -> str:
        """Get global escalation policy."""
        global_config = self.get_global_config()
        return global_config.get("escalation_policy", "on_failure")
    
    def get_max_escalation_attempts(self) -> int:
        """Get global max escalation attempts."""
        global_config = self.get_global_config()
        return global_config.get("max_escalation_attempts", 3)


# ==================== Convenience Functions ====================

def load_noise_config_for_tier(
    tier: str,
    slice_name: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> NoiseConfig:
    """Load noise config for a tier with optional slice overrides.
    
    Args:
        tier: Tier name (fast_noisy, balanced, slow_precise)
        slice_name: Optional slice name for overrides
        config_path: Optional path to config file
    
    Returns:
        NoiseConfig instance
    """
    loader = NoiseConfigLoader(config_path)
    return loader.load_noise_config_for_tier(tier, slice_name)


def load_escalation_config(
    config_path: Optional[Path] = None,
) -> tuple[str, int]:
    """Load escalation policy and max attempts.
    
    Args:
        config_path: Optional path to config file
    
    Returns:
        Tuple of (escalation_policy, max_escalation_attempts)
    """
    loader = NoiseConfigLoader(config_path)
    return loader.get_escalation_policy(), loader.get_max_escalation_attempts()


def is_noise_enabled_for_slice(
    slice_name: str,
    config_path: Optional[Path] = None,
) -> bool:
    """Check if noise is enabled for a slice.
    
    Args:
        slice_name: Slice name
        config_path: Optional path to config file
    
    Returns:
        True if noise is enabled, False otherwise
    """
    loader = NoiseConfigLoader(config_path)
    slice_config = loader.get_slice_config(slice_name)
    
    if slice_config is None:
        # Default to global setting
        global_config = loader.get_global_config()
        return global_config.get("noise_enabled", True)
    
    return slice_config.get("noise_enabled", True)

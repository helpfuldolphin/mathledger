# backend/governance_engine/engine.py
import json
import os
from typing import Dict, Any

from backend.governance_signal.model import GovernanceSignal
from .decision import Decision

# Gracefully handle missing optional dependencies. The engine will enter
# a failsafe mode if they are not installed.
try:
    import yaml
except ImportError:
    yaml = None

try:
    import jsonschema
except ImportError:
    jsonschema = None

POLICY_SCHEMA_FILE = os.path.join(os.path.dirname(__file__), "policy_schema.json")

class GovernanceEngine:
    """
    Loads and validates governance policies. If the policy is invalid or
    dependencies are missing, the engine enters a failsafe 'unstable' state.
    """
    def __init__(self, policy_path: str):
        self.policy_path = policy_path
        self.is_stable = False
        self.rules = []
        self.default_action = "LOG_ADVISORY"
        
        try:
            if yaml is None:
                raise ValueError("`pyyaml` library is not installed.")
            if jsonschema is None:
                raise ValueError("`jsonschema` library is not installed.")

            policy = self._load_policy()
            self._validate_policy(policy)
            
            self.policy = policy
            self.default_action = self.policy.get("default_action", "LOG_ADVISORY")
            self.rules = self.policy.get("rules", [])
            self.is_stable = True
            
        except (ValueError, FileNotFoundError) as e:
            print(f"CRITICAL: GovernanceEngine failed to initialize for '{policy_path}'. Entering UNSTABLE state. Reason: {e}")
            # Engine remains in an unstable state. self.is_stable is False.

    def _load_policy(self) -> Dict[str, Any]:
        """Loads and parses the YAML policy file."""
        try:
            with open(self.policy_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML: {e}")

    def _validate_policy(self, policy: Dict[str, Any]):
        """Validates the loaded policy against the JSON schema."""
        with open(POLICY_SCHEMA_FILE, "r") as f:
            schema = json.load(f)
        try:
            jsonschema.validate(instance=policy, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            raise ValueError(f"Policy is invalid. {e.message}")

    def decide(self, signal: GovernanceSignal) -> Decision:
        """
        Evaluates a signal. If the engine is unstable, it always returns
        a failsafe 'UNSTABLE_POLICY' decision.
        """
        if not self.is_stable:
            return Decision(
                action="UNSTABLE_POLICY",
                triggering_rule="Engine in failsafe mode due to invalid policy or missing dependencies"
            )

        for rule in self.rules:
            if self._rule_matches(rule, signal):
                return Decision(
                    action=rule.get("decision", self.default_action),
                    triggering_rule=rule.get("name", "Unnamed Rule")
                )
        
        return Decision(
            action=self.default_action,
            triggering_rule="Default Action"
        )

    def _rule_matches(self, rule: Dict[str, Any], signal: GovernanceSignal) -> bool:
        """Checks if a signal matches all conditions of a given rule."""
        conditions = rule.get("conditions", {})
        for key, value in conditions.items():
            if not self._condition_met(key, value, signal):
                return False
        return True

    def _condition_met(self, key: str, value: Any, signal: GovernanceSignal) -> bool:
        """Checks a single condition against the signal."""
        signal_value = getattr(signal, key, None)
        if signal_value is None:
            return False

        if isinstance(value, dict):
            min_val = value.get("min")
            max_val = value.get("max")
            if min_val is not None and signal_value < min_val:
                return False
            if max_val is not None and signal_value > max_val:
                return False
            return True
        
        if isinstance(value, list):
            return signal_value in value
        
        return signal_value == value
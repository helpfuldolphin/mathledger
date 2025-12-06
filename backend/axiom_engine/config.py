"""
Axiom configuration system for MathLedger.

Defines axioms and inference rules for propositional logic.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class Axiom:
    """Represents a single axiom."""
    name: str
    content: str
    description: str


@dataclass
class InferenceRule:
    """Represents an inference rule."""
    name: str
    description: str
    arity: int  # Number of premises required


class PropositionalLogicConfig:
    """Configuration for propositional logic axiomatic system."""

    def __init__(self):
        self.axioms = self._define_axioms()
        self.inference_rules = self._define_inference_rules()

    def _define_axioms(self) -> List[Axiom]:
        """Define the axioms for propositional logic."""
        return [
            Axiom(
                name="K",
                content="p -> (q -> p)",
                description="Weakening: if p is true, then q -> p is true for any q"
            ),
            Axiom(
                name="S",
                content="(p -> (q -> r)) -> ((p -> q) -> (p -> r))",
                description="Distribution of implication over implication"
            ),
        ]

    def _define_inference_rules(self) -> List[InferenceRule]:
        """Define the inference rules for propositional logic."""
        return [
            InferenceRule(
                name="modus_ponens",
                description="From p and p -> q, infer q",
                arity=2
            ),
            # Future rules can be added here:
            # InferenceRule(
            #     name="modus_tollens",
            #     description="From p -> q and ~q, infer ~p",
            #     arity=2
            # ),
        ]

    def get_axiom_by_name(self, name: str) -> Axiom | None:
        """Get an axiom by name."""
        for axiom in self.axioms:
            if axiom.name == name:
                return axiom
        return None

    def get_rule_by_name(self, name: str) -> InferenceRule | None:
        """Get an inference rule by name."""
        for rule in self.inference_rules:
            if rule.name == name:
                return rule
        return None


# Global configuration instance
config = PropositionalLogicConfig()

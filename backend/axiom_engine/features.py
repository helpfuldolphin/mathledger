#!/usr/bin/env python3
"""
Fast feature extraction for derivation states and actions.
Extracts numerical features from logical statements and derivation contexts.
"""

import numpy as np
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib


def extract_statement_features(statement: str) -> np.ndarray:
    """
    Extract numerical features from a logical statement.

    Args:
        statement: Logical statement string

    Returns:
        Feature vector of shape (n_features,)
    """
    if not statement:
        return np.zeros(20, dtype=np.float32)

    # Basic text features
    length = len(statement)
    word_count = len(statement.split())
    char_count = len(statement.replace(' ', ''))

    # Logical operator counts
    implies_count = statement.count('->') + statement.count('→')
    and_count = statement.count('/\\') + statement.count('∧') + statement.count('&')
    or_count = statement.count('\\/') + statement.count('∨') + statement.count('|')
    not_count = statement.count('~') + statement.count('¬') + statement.count('!')

    # Parentheses and brackets
    paren_count = statement.count('(') + statement.count(')')
    bracket_count = statement.count('[') + statement.count(']')

    # Variable patterns
    var_pattern = r'\b[a-z]\b'
    variables = set(re.findall(var_pattern, statement.lower()))
    var_count = len(variables)

    # Complexity measures
    nesting_depth = _calculate_nesting_depth(statement)
    operator_density = (implies_count + and_count + or_count + not_count) / max(length, 1)

    # Statement type features
    is_tautology = _is_likely_tautology(statement)
    is_contradiction = _is_likely_contradiction(statement)
    has_quantifiers = any(q in statement for q in ['∀', '∃', 'forall', 'exists'])

    # Hash-based features for uniqueness
    hash_bytes = hashlib.md5(statement.encode()).digest()
    hash_features = [float(b) / 255.0 for b in hash_bytes[:4]]

    # Combine all features
    features = np.array([
        length,
        word_count,
        char_count,
        implies_count,
        and_count,
        or_count,
        not_count,
        paren_count,
        bracket_count,
        var_count,
        nesting_depth,
        operator_density,
        float(is_tautology),
        float(is_contradiction),
        float(has_quantifiers),
    ] + hash_features, dtype=np.float32)

    return features


def extract_context_features(
    state: str,
    action: str,
    available_statements: List[str],
    derivation_depth: int = 0,
    system_id: int = 1
) -> np.ndarray:
    """
    Extract features from derivation context.

    Args:
        state: Current logical state
        action: Proposed derivation action
        available_statements: List of available statements for derivation
        derivation_depth: Current derivation depth
        system_id: Logical system identifier

    Returns:
        Feature vector of shape (n_features,)
    """
    # State features
    state_feats = extract_statement_features(state)

    # Action features
    action_feats = extract_statement_features(action)

    # Context features
    num_available = len(available_statements)
    avg_statement_length = np.mean([len(s) for s in available_statements]) if available_statements else 0

    # Derivation history features
    depth_feature = min(derivation_depth / 10.0, 1.0)  # Normalize to [0,1]

    # System features
    system_feature = float(system_id) / 10.0

    # Interaction features
    state_action_similarity = _calculate_similarity(state, action)
    action_availability = _calculate_action_availability(action, available_statements)

    # Combine all features
    context_features = np.array([
        num_available,
        avg_statement_length,
        depth_feature,
        system_feature,
        state_action_similarity,
        action_availability,
    ], dtype=np.float32)

    # Concatenate all feature vectors
    all_features = np.concatenate([
        state_feats,
        action_feats,
        context_features
    ])

    return all_features


def extract_batch_features(
    states: List[str],
    actions: List[str],
    contexts: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Extract features for a batch of state-action pairs.

    Args:
        states: List of state strings
        actions: List of action strings
        contexts: List of context dictionaries

    Returns:
        Feature matrix of shape (n_samples, n_features)
    """
    if not states or not actions:
        return np.zeros((0, 41), dtype=np.float32)  # 19 + 19 + 3 features

    features_list = []
    for i, (state, action) in enumerate(zip(states, actions)):
        context = contexts[i] if i < len(contexts) else {}

        feats = extract_context_features(
            state=state,
            action=action,
            available_statements=context.get('available_statements', []),
            derivation_depth=context.get('derivation_depth', 0),
            system_id=context.get('system_id', 1)
        )
        features_list.append(feats)

    return np.array(features_list, dtype=np.float32)


def _calculate_nesting_depth(statement: str) -> int:
    """Calculate maximum nesting depth of parentheses."""
    max_depth = 0
    current_depth = 0

    for char in statement:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1

    return max_depth


def _is_likely_tautology(statement: str) -> bool:
    """Heuristic to detect likely tautologies."""
    # Simple patterns that often indicate tautologies
    tautology_patterns = [
        r'p\s*->\s*p',  # p -> p
        r'\([^)]+\)\s*->\s*\1',  # (A) -> A
        r'p\s*\/\\\s*p',  # p /\ p
    ]

    for pattern in tautology_patterns:
        if re.search(pattern, statement, re.IGNORECASE):
            return True

    return False


def _is_likely_contradiction(statement: str) -> bool:
    """Heuristic to detect likely contradictions."""
    # Simple patterns that often indicate contradictions
    contradiction_patterns = [
        r'p\s*/\s*~p',  # p /\ ~p
        r'p\s*/\s*¬p',  # p /\ ¬p
        r'~p\s*/\s*p',  # ~p /\ p
    ]

    for pattern in contradiction_patterns:
        if re.search(pattern, statement, re.IGNORECASE):
            return True

    return False


def _calculate_similarity(str1: str, str2: str) -> float:
    """Calculate simple string similarity."""
    if not str1 or not str2:
        return 0.0

    # Jaccard similarity on character sets
    set1 = set(str1.lower())
    set2 = set(str2.lower())

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def _calculate_action_availability(action: str, available_statements: List[str]) -> float:
    """Calculate how well the action matches available statements."""
    if not available_statements:
        return 0.0

    similarities = [_calculate_similarity(action, stmt) for stmt in available_statements]
    return max(similarities) if similarities else 0.0

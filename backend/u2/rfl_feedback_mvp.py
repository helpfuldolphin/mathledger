"""
RFL Feedback Subsystem MVP

This module provides an MVP implementation of the RFL feedback subsystem.
Implements FeatureExtractor and FeedbackDeriver for 5 key features.

Author: Manus-F
Date: 2025-12-06
Status: Phase V MVP Implementation
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from backend.u2.fosubstrate_executor_skeleton import CandidateItem, ExecutionResult


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

@dataclass
class CandidateFeatures:
    """
    Verifiable features extracted from a candidate and its execution.
    
    MVP includes 5 features:
    1. formula_depth (structural)
    2. mp_depth (search context)
    3. outcome_success (execution)
    4. execution_time_ms (execution)
    5. new_statements_count (execution)
    """
    # Structural features
    formula_depth: int  # Nesting level of formula
    
    # Search features
    mp_depth: int  # Modus Ponens depth in proof tree
    
    # Execution features
    outcome_success: bool  # True if execution succeeded
    execution_time_ms: int  # Time taken
    new_statements_count: int  # Derived statements
    
    def to_feature_vector(self) -> List[float]:
        """Convert to numerical feature vector for ML."""
        return [
            float(self.formula_depth),
            float(self.mp_depth),
            1.0 if self.outcome_success else 0.0,
            float(self.execution_time_ms),
            float(self.new_statements_count),
        ]
    
    @staticmethod
    def feature_names() -> List[str]:
        """Return feature names in vector order."""
        return [
            "formula_depth",
            "mp_depth",
            "outcome_success",
            "execution_time_ms",
            "new_statements_count",
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "formula_depth": self.formula_depth,
            "mp_depth": self.mp_depth,
            "outcome_success": self.outcome_success,
            "execution_time_ms": self.execution_time_ms,
            "new_statements_count": self.new_statements_count,
        }


class FeatureExtractor:
    """
    Extracts verifiable features from candidates and execution results.
    """
    
    def extract(
        self,
        candidate: CandidateItem,
        result: ExecutionResult,
    ) -> CandidateFeatures:
        """
        Extract features from candidate and execution result.
        
        Args:
            candidate: CandidateItem that was executed
            result: ExecutionResult from execution
            
        Returns:
            CandidateFeatures object
        """
        # Extract structural features
        formula_depth = self._compute_formula_depth(candidate.statement.normalized)
        
        # Extract search features
        mp_depth = candidate.statement.mp_depth
        
        # Extract execution features
        outcome_success = (result.outcome.value == "success")
        execution_time_ms = result.time_ms
        new_statements_count = len(result.new_statements)
        
        return CandidateFeatures(
            formula_depth=formula_depth,
            mp_depth=mp_depth,
            outcome_success=outcome_success,
            execution_time_ms=execution_time_ms,
            new_statements_count=new_statements_count,
        )
    
    def _compute_formula_depth(self, normalized: str) -> int:
        """Compute nesting depth of formula."""
        max_depth = 0
        current_depth = 0
        for char in normalized:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        return max_depth


# ============================================================================
# FEEDBACK DERIVATION
# ============================================================================

@dataclass
class CandidateFeedback:
    """
    Aggregated feedback for a single candidate across executions.
    """
    candidate_hash: str
    features: CandidateFeatures
    
    # Aggregated execution statistics
    total_executions: int
    success_count: int
    failure_count: int
    
    success_rate: float  # success_count / total_executions
    avg_execution_time_ms: float
    avg_new_statements: float
    
    # Provenance
    first_seen_cycle: int
    last_seen_cycle: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "candidate_hash": self.candidate_hash,
            "features": self.features.to_dict(),
            "total_executions": self.total_executions,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "avg_new_statements": self.avg_new_statements,
            "first_seen_cycle": self.first_seen_cycle,
            "last_seen_cycle": self.last_seen_cycle,
        }


class FeedbackDeriver:
    """
    Derives per-candidate feedback from execution traces.
    """
    
    def __init__(self):
        self.extractor = FeatureExtractor()
    
    def derive_feedback(
        self,
        trace_path: Path,
    ) -> Dict[str, CandidateFeedback]:
        """
        Derive feedback from execution trace.
        
        Args:
            trace_path: Path to trace JSONL file
            
        Returns:
            Dictionary mapping candidate_hash → CandidateFeedback
        """
        # Read trace
        executions = []
        with open(trace_path, 'r') as f:
            for line in f:
                event = json.loads(line)
                if event["event_type"] == "execution_result":
                    executions.append(event["data"])
        
        # Group by candidate hash
        by_candidate = {}
        for exec_data in executions:
            candidate_hash = exec_data["candidate_hash"]
            if candidate_hash not in by_candidate:
                by_candidate[candidate_hash] = []
            by_candidate[candidate_hash].append(exec_data)
        
        # Aggregate feedback
        feedback = {}
        for candidate_hash, exec_list in by_candidate.items():
            feedback[candidate_hash] = self._aggregate_executions(
                candidate_hash, exec_list
            )
        
        return feedback
    
    def _aggregate_executions(
        self,
        candidate_hash: str,
        executions: List[Dict[str, Any]],
    ) -> CandidateFeedback:
        """Aggregate multiple executions of same candidate."""
        total = len(executions)
        success_count = sum(
            1 for e in executions 
            if e["result"]["outcome"] == "success"
        )
        failure_count = total - success_count
        
        success_rate = success_count / total if total > 0 else 0.0
        
        avg_time = sum(e["result"]["time_ms"] for e in executions) / total
        avg_new_stmts = sum(
            len(e["result"]["new_statements"]) for e in executions
        ) / total
        
        # Extract features from first execution
        first_exec = executions[0]
        features = self._extract_features_from_exec(first_exec)
        
        # Provenance
        first_cycle = min(e.get("cycle", 0) for e in executions)
        last_cycle = max(e.get("cycle", 0) for e in executions)
        
        return CandidateFeedback(
            candidate_hash=candidate_hash,
            features=features,
            total_executions=total,
            success_count=success_count,
            failure_count=failure_count,
            success_rate=success_rate,
            avg_execution_time_ms=avg_time,
            avg_new_statements=avg_new_stmts,
            first_seen_cycle=first_cycle,
            last_seen_cycle=last_cycle,
        )
    
    def _extract_features_from_exec(
        self,
        exec_data: Dict[str, Any],
    ) -> CandidateFeatures:
        """Extract features from execution data."""
        candidate_dict = exec_data["candidate"]
        result_dict = exec_data["result"]
        
        # Reconstruct objects (simplified)
        formula_depth = self._compute_formula_depth(
            candidate_dict["statement"]["normalized"]
        )
        mp_depth = candidate_dict["statement"]["mp_depth"]
        outcome_success = (result_dict["outcome"] == "success")
        execution_time_ms = result_dict["time_ms"]
        new_statements_count = len(result_dict["new_statements"])
        
        return CandidateFeatures(
            formula_depth=formula_depth,
            mp_depth=mp_depth,
            outcome_success=outcome_success,
            execution_time_ms=execution_time_ms,
            new_statements_count=new_statements_count,
        )
    
    def _compute_formula_depth(self, normalized: str) -> int:
        """Compute nesting depth of formula."""
        max_depth = 0
        current_depth = 0
        for char in normalized:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        return max_depth
    
    def save_feedback(
        self,
        feedback: Dict[str, CandidateFeedback],
        output_path: Path,
    ):
        """Save feedback to JSON file."""
        feedback_dict = {
            hash: fb.to_dict()
            for hash, fb in feedback.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(feedback_dict, f, indent=2, sort_keys=True)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def derive_feedback_from_trace(trace_path: Path, output_path: Path):
    """
    Derive feedback from trace and save to file.
    
    Args:
        trace_path: Path to trace JSONL file
        output_path: Path to save feedback JSON
    """
    deriver = FeedbackDeriver()
    
    # Derive feedback
    feedback = deriver.derive_feedback(trace_path)
    
    print(f"Derived feedback for {len(feedback)} candidates")
    
    # Print summary
    for candidate_hash, fb in list(feedback.items())[:5]:  # First 5
        print(f"\nCandidate: {candidate_hash[:16]}...")
        print(f"  Success rate: {fb.success_rate:.2%}")
        print(f"  Avg time: {fb.avg_execution_time_ms:.1f}ms")
        print(f"  Avg new statements: {fb.avg_new_statements:.1f}")
        print(f"  Features: {fb.features.to_feature_vector()}")
    
    # Save feedback
    deriver.save_feedback(feedback, output_path)
    print(f"\nFeedback saved to {output_path}")


if __name__ == "__main__":
    # Example: derive feedback from trace
    trace_path = Path("/tmp/test_trace.jsonl")
    output_path = Path("/tmp/feedback.json")
    
    if trace_path.exists():
        derive_feedback_from_trace(trace_path, output_path)
    else:
        print(f"Trace not found: {trace_path}")
        print("Run distributed_frontier_mvp.py first to generate a trace.")

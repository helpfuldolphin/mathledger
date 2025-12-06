#!/usr/bin/env python3
"""
Meta-Scheduler for CI Pipeline Optimization
Analyzes job DAGs and reorders steps to minimize wall-clock latency
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque


class JobDAG:
    """Represents a Directed Acyclic Graph of CI jobs"""
    
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, List[str]] = defaultdict(list)  # job -> dependencies
        self.reverse_edges: Dict[str, List[str]] = defaultdict(list)  # job -> dependents
        self.job_durations: Dict[str, float] = {}
    
    def add_job(self, job_name: str, duration: float = 0):
        """Add a job to the DAG"""
        self.nodes.add(job_name)
        self.job_durations[job_name] = duration
    
    def add_dependency(self, job: str, depends_on: str):
        """Add a dependency: job depends on depends_on"""
        self.edges[job].append(depends_on)
        self.reverse_edges[depends_on].append(job)
    
    def topological_sort(self) -> List[str]:
        """Return jobs in topological order"""
        in_degree = {node: 0 for node in self.nodes}
        for node in self.nodes:
            for dep in self.edges[node]:
                in_degree[node] += 1
        
        queue = deque([node for node in self.nodes if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for dependent in self.reverse_edges[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(result) != len(self.nodes):
            raise ValueError("Cycle detected in job DAG")
        
        return result
    
    def get_parallelizable_jobs(self) -> List[Set[str]]:
        """Return sets of jobs that can run in parallel"""
        levels = []
        processed = set()
        
        while len(processed) < len(self.nodes):
            # Find jobs whose dependencies are all processed
            current_level = set()
            for node in self.nodes:
                if node in processed:
                    continue
                if all(dep in processed for dep in self.edges[node]):
                    current_level.add(node)
            
            if not current_level:
                raise ValueError("Cycle detected or invalid DAG")
            
            levels.append(current_level)
            processed.update(current_level)
        
        return levels
    
    def calculate_critical_path(self) -> Tuple[List[str], float]:
        """Calculate the critical path (longest path) through the DAG"""
        topo_order = self.topological_sort()
        
        # Calculate earliest start times
        earliest_start = {node: 0 for node in self.nodes}
        for node in topo_order:
            for dep in self.edges[node]:
                earliest_start[node] = max(
                    earliest_start[node],
                    earliest_start[dep] + self.job_durations[dep]
                )
        
        # Find the critical path by backtracking from the longest job
        max_node = max(self.nodes, key=lambda n: earliest_start[n] + self.job_durations[n])
        critical_path = [max_node]
        current = max_node
        
        while self.edges[current]:
            # Find the dependency that contributes to the critical path
            next_node = max(
                self.edges[current],
                key=lambda n: earliest_start[n] + self.job_durations[n]
            )
            critical_path.append(next_node)
            current = next_node
        
        critical_path.reverse()
        total_time = earliest_start[max_node] + self.job_durations[max_node]
        
        return critical_path, total_time


class MetaScheduler:
    """Optimizes CI workflow execution order"""
    
    def __init__(self, workflow_path: str):
        self.workflow_path = Path(workflow_path)
        self.workflow_data = self._load_workflow()
        self.dag = self._build_dag()
    
    def _load_workflow(self) -> Dict:
        """Load workflow YAML"""
        with open(self.workflow_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _build_dag(self) -> JobDAG:
        """Build job DAG from workflow definition"""
        dag = JobDAG()
        jobs = self.workflow_data.get('jobs', {})
        
        for job_name, job_config in jobs.items():
            # Estimate duration (would be replaced with telemetry data)
            duration = self._estimate_job_duration(job_name, job_config)
            dag.add_job(job_name, duration)
            
            # Add dependencies
            needs = job_config.get('needs', [])
            if isinstance(needs, str):
                needs = [needs]
            for dep in needs:
                dag.add_dependency(job_name, dep)
        
        return dag
    
    def _estimate_job_duration(self, job_name: str, job_config: Dict) -> float:
        """Estimate job duration based on steps"""
        # Simple heuristic: count steps and estimate
        steps = job_config.get('steps', [])
        base_duration = 30  # Base overhead
        step_duration = len(steps) * 15  # 15s per step average
        return base_duration + step_duration
    
    def optimize_execution_order(self) -> Dict[str, Any]:
        """Generate optimized execution plan"""
        try:
            parallel_levels = self.dag.get_parallelizable_jobs()
            critical_path, critical_time = self.dag.calculate_critical_path()
            
            optimization = {
                'parallel_levels': [list(level) for level in parallel_levels],
                'critical_path': critical_path,
                'estimated_critical_time': critical_time,
                'parallelization_opportunities': len(parallel_levels),
                'max_parallel_jobs': max(len(level) for level in parallel_levels),
                'recommendations': self._generate_recommendations(parallel_levels, critical_path)
            }
            
            return optimization
        except ValueError as e:
            return {'error': str(e)}
    
    def _generate_recommendations(self, parallel_levels: List[Set[str]], 
                                  critical_path: List[str]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Check for underutilized parallelism
        for i, level in enumerate(parallel_levels):
            if len(level) == 1 and i < len(parallel_levels) - 1:
                recommendations.append(
                    f"Level {i}: Single job {list(level)[0]} - consider splitting or merging"
                )
        
        # Highlight critical path jobs
        for job in critical_path:
            if self.dag.job_durations[job] > 120:
                recommendations.append(
                    f"Critical path job '{job}' takes {self.dag.job_durations[job]:.0f}s - optimize this for maximum impact"
                )
        
        # Check for excessive parallelism
        max_parallel = max(len(level) for level in parallel_levels)
        if max_parallel > 4:
            recommendations.append(
                f"Max {max_parallel} parallel jobs may exceed runner capacity - consider batching"
            )
        
        return recommendations
    
    def apply_telemetry_durations(self, telemetry_path: str):
        """Update job durations from telemetry data"""
        telemetry_file = Path(telemetry_path)
        if not telemetry_file.exists():
            return
        
        with open(telemetry_file, 'r') as f:
            data = json.load(f)
        
        # Extract average durations from recent runs
        job_durations = defaultdict(list)
        for run in data.get('runs', [])[-10:]:  # Last 10 runs
            for job in run.get('jobs', []):
                job_durations[job['job_name']].append(job['duration_seconds'])
        
        # Update DAG with average durations
        for job_name, durations in job_durations.items():
            if job_name in self.dag.nodes:
                avg_duration = sum(durations) / len(durations)
                self.dag.job_durations[job_name] = avg_duration


def analyze_workflow(workflow_path: str, telemetry_path: Optional[str] = None) -> Dict[str, Any]:
    """Analyze a workflow and generate optimization report"""
    scheduler = MetaScheduler(workflow_path)
    
    if telemetry_path:
        scheduler.apply_telemetry_durations(telemetry_path)
    
    optimization = scheduler.optimize_execution_order()
    
    return {
        'workflow': str(workflow_path),
        'optimization': optimization,
        'dag_stats': {
            'total_jobs': len(scheduler.dag.nodes),
            'total_dependencies': sum(len(deps) for deps in scheduler.dag.edges.values()),
            'estimated_total_duration': sum(scheduler.dag.job_durations.values())
        }
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: meta_scheduler.py <workflow.yml> [telemetry.json]")
        sys.exit(1)
    
    workflow_path = sys.argv[1]
    telemetry_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = analyze_workflow(workflow_path, telemetry_path)
    print(json.dumps(result, indent=2))

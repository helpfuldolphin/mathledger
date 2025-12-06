#!/usr/bin/env python3
"""
Velocity Telemetry System for CI Pipeline Orchestration
Collects timing data and generates performance metrics for meta-scheduling
"""

import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class JobTiming:
    """Timing data for a single CI job"""
    job_name: str
    start_time: float
    end_time: float
    duration_seconds: float
    status: str  # pass, fail, skip
    step_timings: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowRun:
    """Complete workflow run with all job timings"""
    workflow_name: str
    run_id: str
    trigger: str  # push, pull_request, workflow_dispatch
    start_time: float
    end_time: float
    total_duration_seconds: float
    jobs: List[JobTiming]
    optimization_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'workflow_name': self.workflow_name,
            'run_id': self.run_id,
            'trigger': self.trigger,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_duration_seconds': self.total_duration_seconds,
            'jobs': [j.to_dict() for j in self.jobs],
            'optimization_hash': self.optimization_hash
        }


class VelocityTelemetry:
    """Collects and analyzes CI pipeline timing data"""
    
    def __init__(self, output_path: str = "artifacts/ci/perf_log.json"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs: List[WorkflowRun] = []
        self._load_existing()
    
    def _load_existing(self):
        """Load existing telemetry data"""
        if self.output_path.exists():
            try:
                with open(self.output_path, 'r') as f:
                    data = json.load(f)
                    # Load runs from canonical format
                    for run_data in data.get('runs', []):
                        jobs = [JobTiming(**j) for j in run_data['jobs']]
                        run = WorkflowRun(
                            workflow_name=run_data['workflow_name'],
                            run_id=run_data['run_id'],
                            trigger=run_data['trigger'],
                            start_time=run_data['start_time'],
                            end_time=run_data['end_time'],
                            total_duration_seconds=run_data['total_duration_seconds'],
                            jobs=jobs,
                            optimization_hash=run_data['optimization_hash']
                        )
                        self.runs.append(run)
            except (json.JSONDecodeError, KeyError):
                # Invalid or corrupted data, start fresh
                self.runs = []
    
    def record_run(self, workflow_run: WorkflowRun):
        """Record a workflow run"""
        self.runs.append(workflow_run)
        self._save()
    
    def _save(self):
        """Save telemetry data in canonical JSON format"""
        data = {
            'version': '1.0',
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'runs': [run.to_dict() for run in self.runs]
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def calculate_baseline(self, workflow_name: str, lookback: int = 10) -> Optional[float]:
        """Calculate baseline duration for a workflow"""
        relevant_runs = [
            r for r in self.runs[-lookback:]
            if r.workflow_name == workflow_name
        ]
        
        if not relevant_runs:
            return None
        
        return sum(r.total_duration_seconds for r in relevant_runs) / len(relevant_runs)
    
    def calculate_velocity_improvement(self, workflow_name: str, current_duration: float) -> Optional[float]:
        """Calculate velocity improvement percentage"""
        baseline = self.calculate_baseline(workflow_name)
        if baseline is None or baseline == 0:
            return None
        
        improvement = ((baseline - current_duration) / baseline) * 100
        return improvement
    
    def generate_optimization_hash(self, workflow_config: Dict[str, Any]) -> str:
        """Generate SHA256 hash of workflow optimization configuration"""
        # Sort keys for deterministic hashing
        config_str = json.dumps(workflow_config, sort_keys=True)
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()
    
    def get_job_dag(self, workflow_name: str) -> Dict[str, List[str]]:
        """Extract job dependency DAG from recent runs"""
        # Analyze recent runs to determine job dependencies
        recent_run = next((r for r in reversed(self.runs) if r.workflow_name == workflow_name), None)
        
        if not recent_run:
            return {}
        
        # Build DAG based on job execution order
        dag = {}
        for job in recent_run.jobs:
            dag[job.job_name] = []  # Dependencies would be extracted from workflow YAML
        
        return dag
    
    def suggest_optimizations(self, workflow_name: str) -> List[str]:
        """Suggest optimizations based on telemetry data"""
        suggestions = []
        
        recent_runs = [r for r in self.runs[-5:] if r.workflow_name == workflow_name]
        if not recent_runs:
            return suggestions
        
        # Analyze job durations
        job_durations = {}
        for run in recent_runs:
            for job in run.jobs:
                if job.job_name not in job_durations:
                    job_durations[job.job_name] = []
                job_durations[job.job_name].append(job.duration_seconds)
        
        # Find slow jobs
        for job_name, durations in job_durations.items():
            avg_duration = sum(durations) / len(durations)
            if avg_duration > 120:  # Jobs taking >2 minutes
                suggestions.append(f"Optimize {job_name}: avg {avg_duration:.1f}s")
        
        # Check for sequential jobs that could be parallelized
        if len(job_durations) > 1:
            suggestions.append("Consider parallelizing independent jobs")
        
        return suggestions


def create_ci_summary(workflow_name: str, current_duration: float, 
                     optimization_hash: str, telemetry: VelocityTelemetry) -> str:
    """Generate CI summary with velocity metrics"""
    improvement = telemetry.calculate_velocity_improvement(workflow_name, current_duration)
    
    if improvement is None:
        status = "[BASELINE] CI Velocity: Establishing baseline"
    elif improvement >= 0:
        status = f"[PASS] CI Velocity: {improvement:.1f}% faster"
    else:
        status = f"[REGRESSION] CI Velocity: {abs(improvement):.1f}% slower"
    
    summary = f"{status}\nCI_OPTIMIZATION_HASH: {optimization_hash}\n"
    return summary


if __name__ == "__main__":
    # Example usage
    telemetry = VelocityTelemetry()
    
    # Simulate a workflow run
    job1 = JobTiming(
        job_name="test",
        start_time=time.time(),
        end_time=time.time() + 120,
        duration_seconds=120,
        status="pass",
        step_timings=[]
    )
    
    job2 = JobTiming(
        job_name="uplift-omega",
        start_time=time.time() + 120,
        end_time=time.time() + 180,
        duration_seconds=60,
        status="pass",
        step_timings=[]
    )
    
    workflow_config = {
        "caching": True,
        "parallel_jobs": ["test", "uplift-omega"],
        "optimization_level": 2
    }
    
    run = WorkflowRun(
        workflow_name="ci.yml",
        run_id="test-run-001",
        trigger="pull_request",
        start_time=time.time(),
        end_time=time.time() + 180,
        total_duration_seconds=180,
        jobs=[job1, job2],
        optimization_hash=telemetry.generate_optimization_hash(workflow_config)
    )
    
    telemetry.record_run(run)
    
    summary = create_ci_summary("ci.yml", 180, run.optimization_hash, telemetry)
    print(summary)

#!/usr/bin/env python3
"""
Benchmark Orchestration Runner

Executes the 5-week benchmark plan defined in orchestration.yaml.

Author: Manus-H
"""

import os
import sys
import yaml
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class BenchmarkOrchestrator:
    """Orchestrates benchmark execution according to YAML plan."""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = self.load_config()
        self.output_dir = Path(self.config["config"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def load_config(self) -> Dict:
        """Load orchestration configuration from YAML."""
        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def run_week(self, week_name: str) -> Dict[str, Any]:
        """
        Run all tasks for a specific week.
        
        Args:
            week_name: Name of the week (e.g., "week1")
            
        Returns:
            Dictionary of results
        """
        week_config = self.config[week_name]
        print(f"\n{'='*80}")
        print(f"Starting {week_config['name']}")
        print(f"Description: {week_config['description']}")
        print(f"{'='*80}\n")
        
        week_results = {
            "name": week_config["name"],
            "start_time": datetime.now().isoformat(),
            "tasks": {},
            "acceptance_criteria": {},
        }
        
        # Execute tasks
        for task in week_config["tasks"]:
            task_name = task["name"]
            print(f"Running task: {task_name}")
            
            # Check dependencies
            if "depends_on" in task:
                for dep in task["depends_on"]:
                    if dep not in week_results["tasks"]:
                        print(f"  Skipping {task_name}: dependency {dep} not completed")
                        continue
            
            # Execute task
            result = self.run_task(task)
            week_results["tasks"][task_name] = result
            
            if result["success"]:
                print(f"  ✓ {task_name} completed successfully")
            else:
                print(f"  ✗ {task_name} failed: {result['error']}")
                
                # Check if task is optional
                if not task.get("optional", False):
                    print(f"  Critical task failed, aborting week")
                    break
        
        # Check acceptance criteria
        if "acceptance_criteria" in week_config:
            print(f"\nChecking acceptance criteria for {week_config['name']}...")
            acceptance_results = self.check_acceptance_criteria(
                week_config["acceptance_criteria"],
                week_results["tasks"]
            )
            week_results["acceptance_criteria"] = acceptance_results
            
            # Report acceptance status
            passed = sum(1 for r in acceptance_results.values() if r["passed"])
            total = len(acceptance_results)
            print(f"  Acceptance: {passed}/{total} criteria passed")
        
        week_results["end_time"] = datetime.now().isoformat()
        
        # Generate reports
        if "reports" in week_config:
            print(f"\nGenerating reports for {week_config['name']}...")
            for report in week_config["reports"]:
                self.generate_report(report, week_results)
        
        return week_results
    
    def run_task(self, task: Dict) -> Dict[str, Any]:
        """
        Run a single benchmark task.
        
        Args:
            task: Task configuration dictionary
            
        Returns:
            Task result dictionary
        """
        command = task["command"]
        output_file = self.output_dir / task["output"]
        
        result = {
            "command": command,
            "output_file": str(output_file),
            "start_time": datetime.now().isoformat(),
            "success": False,
            "error": None,
        }
        
        try:
            # Execute command
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            result["return_code"] = process.returncode
            result["stdout"] = process.stdout
            result["stderr"] = process.stderr
            
            if process.returncode == 0:
                result["success"] = True
            else:
                result["error"] = f"Command failed with return code {process.returncode}"
        
        except subprocess.TimeoutExpired:
            result["error"] = "Command timed out after 1 hour"
        except Exception as e:
            result["error"] = str(e)
        
        result["end_time"] = datetime.now().isoformat()
        
        return result
    
    def check_acceptance_criteria(
        self,
        criteria: List[Dict],
        task_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check acceptance criteria against task results.
        
        Args:
            criteria: List of acceptance criteria
            task_results: Results from completed tasks
            
        Returns:
            Dictionary of acceptance results
        """
        acceptance_results = {}
        
        for criterion in criteria:
            metric = criterion["metric"]
            threshold = criterion["threshold"]
            severity = criterion["severity"]
            
            # Extract metric value from task results
            # In production, this would parse CSV/JSON outputs
            # For now, we'll simulate the check
            metric_value = self.extract_metric_value(metric, task_results)
            
            # Evaluate threshold
            passed = self.evaluate_threshold(metric_value, threshold)
            
            acceptance_results[metric] = {
                "metric": metric,
                "threshold": threshold,
                "actual_value": metric_value,
                "passed": passed,
                "severity": severity,
            }
        
        return acceptance_results
    
    def extract_metric_value(self, metric: str, task_results: Dict) -> float:
        """
        Extract metric value from task results.
        
        This is a placeholder that would parse actual CSV/JSON outputs.
        """
        # TODO: Implement actual metric extraction from CSV/JSON files
        return 0.0
    
    def evaluate_threshold(self, value: float, threshold: str) -> bool:
        """
        Evaluate if a value meets a threshold condition.
        
        Args:
            value: Actual metric value
            threshold: Threshold expression (e.g., ">= 100", "<= 300")
            
        Returns:
            True if threshold is met, False otherwise
        """
        # Parse threshold expression
        if ">=" in threshold:
            threshold_value = float(threshold.split(">=")[1].strip())
            return value >= threshold_value
        elif "<=" in threshold:
            threshold_value = float(threshold.split("<=")[1].strip())
            return value <= threshold_value
        elif ">" in threshold:
            threshold_value = float(threshold.split(">")[1].strip())
            return value > threshold_value
        elif "<" in threshold:
            threshold_value = float(threshold.split("<")[1].strip())
            return value < threshold_value
        else:
            return False
    
    def generate_report(self, report_config: Dict, week_results: Dict) -> None:
        """
        Generate a report from week results.
        
        Args:
            report_config: Report configuration
            week_results: Results from the week
        """
        report_name = report_config["name"]
        output_file = self.output_dir / report_config["output"]
        
        print(f"  Generating report: {report_name}")
        
        # Generate markdown report
        report_content = self.format_report_markdown(report_name, week_results)
        
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        print(f"  Report saved to: {output_file}")
    
    def format_report_markdown(self, report_name: str, week_results: Dict) -> str:
        """Format week results as markdown report."""
        lines = []
        
        lines.append(f"# {report_name}")
        lines.append(f"\n**Generated**: {datetime.now().isoformat()}")
        lines.append(f"\n**Week**: {week_results['name']}")
        lines.append(f"\n**Duration**: {week_results['start_time']} to {week_results['end_time']}")
        
        # Task results
        lines.append("\n## Task Results\n")
        for task_name, task_result in week_results["tasks"].items():
            status = "✓" if task_result["success"] else "✗"
            lines.append(f"- {status} **{task_name}**")
            if not task_result["success"]:
                lines.append(f"  - Error: {task_result['error']}")
        
        # Acceptance criteria
        if week_results["acceptance_criteria"]:
            lines.append("\n## Acceptance Criteria\n")
            for metric, result in week_results["acceptance_criteria"].items():
                status = "✓ PASS" if result["passed"] else "✗ FAIL"
                lines.append(f"- {status} **{metric}**")
                lines.append(f"  - Threshold: {result['threshold']}")
                lines.append(f"  - Actual: {result['actual_value']}")
                lines.append(f"  - Severity: {result['severity']}")
        
        return "\n".join(lines)
    
    def run_all_weeks(self) -> Dict[str, Any]:
        """
        Run all weeks in sequence.
        
        Returns:
            Dictionary of all results
        """
        all_results = {
            "start_time": datetime.now().isoformat(),
            "weeks": {},
        }
        
        # Run each week
        for week_name in ["week1", "week2", "week3", "week4", "week5"]:
            week_results = self.run_week(week_name)
            all_results["weeks"][week_name] = week_results
            
            # Save intermediate results
            self.save_results(all_results)
        
        # Generate final report
        print(f"\n{'='*80}")
        print("Generating final report...")
        print(f"{'='*80}\n")
        
        final_report = self.generate_final_report(all_results)
        all_results["final_report"] = final_report
        
        all_results["end_time"] = datetime.now().isoformat()
        
        # Save final results
        self.save_results(all_results)
        
        return all_results
    
    def generate_final_report(self, all_results: Dict) -> Dict:
        """Generate final summary report."""
        final_report = {
            "summary": {
                "total_weeks": len(all_results["weeks"]),
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "acceptance_criteria_passed": 0,
                "acceptance_criteria_failed": 0,
            },
            "recommendations": [],
        }
        
        # Aggregate statistics
        for week_name, week_results in all_results["weeks"].items():
            for task_name, task_result in week_results["tasks"].items():
                final_report["summary"]["total_tasks"] += 1
                if task_result["success"]:
                    final_report["summary"]["successful_tasks"] += 1
                else:
                    final_report["summary"]["failed_tasks"] += 1
            
            for metric, result in week_results.get("acceptance_criteria", {}).items():
                if result["passed"]:
                    final_report["summary"]["acceptance_criteria_passed"] += 1
                else:
                    final_report["summary"]["acceptance_criteria_failed"] += 1
        
        # Generate recommendations
        if final_report["summary"]["acceptance_criteria_failed"] > 0:
            final_report["recommendations"].append(
                "Some acceptance criteria failed. Review failed criteria and consider optimizations."
            )
        
        if final_report["summary"]["failed_tasks"] > 0:
            final_report["recommendations"].append(
                f"{final_report['summary']['failed_tasks']} tasks failed. Investigate failures before deployment."
            )
        
        # Write final report
        final_report_file = self.output_dir / "final_benchmark_report.md"
        report_content = self.format_final_report_markdown(final_report, all_results)
        
        with open(final_report_file, 'w') as f:
            f.write(report_content)
        
        print(f"Final report saved to: {final_report_file}")
        
        return final_report
    
    def format_final_report_markdown(self, final_report: Dict, all_results: Dict) -> str:
        """Format final report as markdown."""
        lines = []
        
        lines.append("# PQ Migration Benchmark - Final Report")
        lines.append(f"\n**Generated**: {datetime.now().isoformat()}")
        lines.append(f"\n**Duration**: {all_results['start_time']} to {all_results['end_time']}")
        
        # Summary
        lines.append("\n## Executive Summary\n")
        summary = final_report["summary"]
        lines.append(f"- **Total Weeks**: {summary['total_weeks']}")
        lines.append(f"- **Total Tasks**: {summary['total_tasks']}")
        lines.append(f"- **Successful Tasks**: {summary['successful_tasks']}")
        lines.append(f"- **Failed Tasks**: {summary['failed_tasks']}")
        lines.append(f"- **Acceptance Criteria Passed**: {summary['acceptance_criteria_passed']}")
        lines.append(f"- **Acceptance Criteria Failed**: {summary['acceptance_criteria_failed']}")
        
        # Recommendations
        if final_report["recommendations"]:
            lines.append("\n## Recommendations\n")
            for rec in final_report["recommendations"]:
                lines.append(f"- {rec}")
        
        # Week-by-week results
        lines.append("\n## Week-by-Week Results\n")
        for week_name, week_results in all_results["weeks"].items():
            lines.append(f"\n### {week_results['name']}\n")
            
            successful = sum(1 for t in week_results["tasks"].values() if t["success"])
            total = len(week_results["tasks"])
            lines.append(f"- Tasks: {successful}/{total} successful")
            
            if week_results.get("acceptance_criteria"):
                passed = sum(1 for r in week_results["acceptance_criteria"].values() if r["passed"])
                total_criteria = len(week_results["acceptance_criteria"])
                lines.append(f"- Acceptance: {passed}/{total_criteria} passed")
        
        return "\n".join(lines)
    
    def save_results(self, results: Dict) -> None:
        """Save results to JSON file."""
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python orchestrate_benchmarks.py [week1|week2|week3|week4|week5|all]")
        sys.exit(1)
    
    target = sys.argv[1]
    
    # Initialize orchestrator
    orchestrator = BenchmarkOrchestrator("backend/benchmarks/orchestration.yaml")
    
    if target == "all":
        # Run all weeks
        results = orchestrator.run_all_weeks()
    elif target in ["week1", "week2", "week3", "week4", "week5"]:
        # Run specific week
        results = orchestrator.run_week(target)
    else:
        print(f"Unknown target: {target}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("Benchmark orchestration complete!")
    print(f"Results saved to: {orchestrator.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

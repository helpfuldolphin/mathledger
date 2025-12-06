#!/usr/bin/env python3
"""
Real-Time Metrics API
Provides HTTP endpoints for monitoring dashboard data.

Endpoints:
- GET /metrics - Current metrics snapshot
- GET /metrics/proof - Proof generation metrics
- GET /metrics/curriculum - Curriculum status
- GET /metrics/agents - Agent coordination status
- GET /metrics/ci - CI/CD health
- GET /health - Health check
"""
from flask import Flask, jsonify
from monitoring_dashboard import Dashboard
import os

app = Flask(__name__)

# Initialize dashboard
artifacts_dir = os.environ.get('ARTIFACTS_DIR', 'artifacts')
dashboard = Dashboard(artifacts_dir=artifacts_dir)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'mathledger-metrics-api',
        'version': 'v1'
    })


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get complete metrics snapshot"""
    snapshot = dashboard.collect_snapshot()
    
    return jsonify({
        'timestamp': snapshot.timestamp,
        'proof_metrics': {
            'total_proofs': snapshot.proof_metrics.total_proofs,
            'total_blocks': snapshot.proof_metrics.total_blocks,
            'avg_proofs_per_block': snapshot.proof_metrics.avg_proofs_per_block,
            'success_rate': snapshot.proof_metrics.success_rate,
            'throughput_proofs_per_hour': snapshot.proof_metrics.throughput_proofs_per_hour
        },
        'curriculum_status': {
            'current_slice': snapshot.curriculum_status.current_slice,
            'total_proofs': snapshot.curriculum_status.total_proofs,
            'threshold': snapshot.curriculum_status.threshold,
            'progress_pct': snapshot.curriculum_status.progress_pct,
            'blocks_sealed': snapshot.curriculum_status.blocks_sealed,
            'status': snapshot.curriculum_status.status
        },
        'agent_status': {
            'total_agents': snapshot.agent_status.total_agents,
            'active_agents': snapshot.agent_status.active_agents,
            'ready_agents': snapshot.agent_status.ready_agents,
            'dormant_agents': snapshot.agent_status.dormant_agents,
            'open_prs': snapshot.agent_status.open_prs
        },
        'ci_health': {
            'total_workflows': snapshot.ci_health.total_workflows,
            'passing_workflows': snapshot.ci_health.passing_workflows,
            'failing_workflows': snapshot.ci_health.failing_workflows,
            'avg_runtime_minutes': snapshot.ci_health.avg_runtime_minutes
        },
        'alerts': snapshot.alerts
    })


@app.route('/metrics/proof', methods=['GET'])
def get_proof_metrics():
    """Get proof generation metrics only"""
    metrics = dashboard.collector.collect_proof_metrics()
    return jsonify({
        'total_proofs': metrics.total_proofs,
        'total_blocks': metrics.total_blocks,
        'avg_proofs_per_block': metrics.avg_proofs_per_block,
        'success_rate': metrics.success_rate,
        'throughput_proofs_per_hour': metrics.throughput_proofs_per_hour,
        'last_block_time': metrics.last_block_time
    })


@app.route('/metrics/curriculum', methods=['GET'])
def get_curriculum_status():
    """Get curriculum status only"""
    status = dashboard.collector.collect_curriculum_status()
    return jsonify({
        'current_slice': status.current_slice,
        'total_proofs': status.total_proofs,
        'threshold': status.threshold,
        'progress_pct': status.progress_pct,
        'blocks_sealed': status.blocks_sealed,
        'status': status.status
    })


@app.route('/metrics/agents', methods=['GET'])
def get_agent_status():
    """Get agent coordination status only"""
    status = dashboard.collector.collect_agent_status()
    return jsonify({
        'total_agents': status.total_agents,
        'active_agents': status.active_agents,
        'ready_agents': status.ready_agents,
        'dormant_agents': status.dormant_agents,
        'open_prs': status.open_prs
    })


@app.route('/metrics/ci', methods=['GET'])
def get_ci_health():
    """Get CI/CD health only"""
    health = dashboard.collector.collect_ci_health()
    return jsonify({
        'total_workflows': health.total_workflows,
        'passing_workflows': health.passing_workflows,
        'failing_workflows': health.failing_workflows,
        'last_run_time': health.last_run_time,
        'avg_runtime_minutes': health.avg_runtime_minutes
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


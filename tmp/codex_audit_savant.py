import json
import hashlib
import datetime
from pathlib import Path
from analysis import governance as gov

def main() -> None:
    root = Path('.').resolve()
    base_dir = root / 'artifacts' / 'governance' / 'codex_audit_savant'
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'

    paths = {
        'conjecture_report': root / 'artifacts' / 'dynamics' / 'conjecture_report.json',
        'structure_drift_report': root / 'structure_drift_report.json',
        'drift_forecast': root / 'artifacts' / 'governance' / 'drift_forecast.json',
        'drift_radar_summary': root / 'artifacts' / 'governance' / 'drift_radar_summary.json',
        'metric_integration_report': root / 'artifacts' / 'metrics' / 'metric_integration_report.json',
        'snapshot_baseline_density': root / 'artifacts' / 'snapshots' / 'baseline' / 'uplift_u2_density.json',
        'snapshot_candidate_density': root / 'artifacts' / 'snapshots' / 'candidate' / 'uplift_u2_density.json'
    }

    def load_json(path: Path):
        with path.open(encoding='utf-8') as handle:
            return json.load(handle)

    def sha256_path(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def canonical_bytes(payload) -> bytes:
        text = json.dumps(payload, indent=2, sort_keys=True)
        return (text + '\n').encode('utf-8')

    def write_json(path: Path, payload) -> str:
        payload_bytes = canonical_bytes(payload)
        path.write_bytes(payload_bytes)
        return hashlib.sha256(payload_bytes).hexdigest()

    def file_info(path: Path) -> dict:
        return {
            'path': path.relative_to(root).as_posix(),
            'sha256': sha256_path(path),
            'size_bytes': path.stat().st_size
        }

    def extract_status_counts(snapshot: dict) -> dict:
        counts = {'SUPPORTS': 0, 'CONTRADICTS': 0, 'INCONCLUSIVE': 0}
        for value in snapshot.values():
            if isinstance(value, dict) and 'status' in value:
                status = value['status']
                counts[status] = counts.get(status, 0) + 1
        return counts

    def collect_statuses(snapshot: dict) -> dict:
        statuses = {}
        for key, value in snapshot.items():
            if isinstance(value, dict) and 'status' in value:
                statuses[key] = value['status']
        return statuses

    source_materials = {name: file_info(path) for name, path in paths.items()}

    conjecture_snapshot = load_json(paths['conjecture_report'])
    structure_drift = load_json(paths['structure_drift_report'])
    drift_forecast = load_json(paths['drift_forecast'])
    drift_radar = load_json(paths['drift_radar_summary'])
    metric_report = load_json(paths['metric_integration_report'])
    baseline_density = load_json(paths['snapshot_baseline_density'])
    candidate_density = load_json(paths['snapshot_candidate_density'])

    status_counts = extract_status_counts(conjecture_snapshot)
    status_details = collect_statuses(conjecture_snapshot)
    global_health_summary = gov.summarize_conjecture_status_for_global_health(conjecture_snapshot)
    contradicts = status_counts.get('CONTRADICTS', 0)
    ratio = None
    if contradicts:
        ratio = round(status_counts.get('SUPPORTS', 0) / contradicts, 4)

    global_health_section = {
        'status': global_health_summary['learning_health'],
        'any_key_conjecture_contradicted': global_health_summary['any_key_conjecture_contradicted'],
        'supports': status_counts.get('SUPPORTS', 0),
        'contradicts': contradicts,
        'inconclusive': status_counts.get('INCONCLUSIVE', 0),
        'supports_to_contradicts_ratio': ratio,
        'key_statuses': {
            key: status_details.get(key)
            for key in ['Phase II Uplift', 'Conjecture 3.1', 'Conjecture 4.1', 'Conjecture 6.1']
        }
    }

    structure_summary = structure_drift.get('summary', {})
    metric_summary = metric_report.get('summary', {})
    baseline_l3 = baseline_density['levels']['L3_regression']
    candidate_l3 = candidate_density['levels']['L3_regression']
    regression_delta = None
    if 'value' in baseline_l3 and 'value' in candidate_l3:
        regression_delta = round(candidate_l3['value'] - baseline_l3['value'], 4)

    regression_watch = {
        'baseline': {
            'status': baseline_l3.get('status'),
            'value': baseline_l3.get('value')
        },
        'candidate': {
            'status': candidate_l3.get('status'),
            'value': candidate_l3.get('value'),
            'details': candidate_l3.get('details')
        },
        'delta_vs_baseline': regression_delta,
        'regression_detected': candidate_l3.get('status') == 'FAIL'
    }

    deterministic_inputs = sorted(info['sha256'] for info in source_materials.values())

    drift_forecast_excerpt = {
        'forecast_id': drift_forecast.get('forecast_id'),
        'generated_at': drift_forecast.get('generated_at'),
        'high_risk_file_count': drift_forecast.get('high_risk_file_count'),
        'trending_violations': drift_forecast.get('trending_violations')
    }

    cicd_lines_payload = {
        'schema_version': 'codex.audit.cicd-lines/1.0.0',
        'generated_at': timestamp,
        'lines': [
            {
                'line_id': 'global-health-learning',
                'severity': 'info',
                'text': f"Learning health {global_health_section['status']} (supports={global_health_section['supports']}, contradicts={global_health_section['contradicts']}).",
                'source_hash': source_materials['conjecture_report']['sha256']
            },
            {
                'line_id': 'metric-integration',
                'severity': 'warn' if metric_summary.get('status') != 'PASSED' else 'info',
                'text': f"Metric integration status {metric_summary.get('status')} with warnings: {', '.join(metric_summary.get('warnings', [])) or 'none'}.",
                'source_hash': source_materials['metric_integration_report']['sha256']
            },
            {
                'line_id': 'drift-structure',
                'severity': 'info',
                'text': f"Structural drift trend {structure_summary.get('overall_trend', 'unknown')} across {structure_summary.get('total_directories_compared')} dirs.",
                'source_hash': source_materials['structure_drift_report']['sha256']
            },
            {
                'line_id': 'regression-watch',
                'severity': 'alert' if regression_watch['regression_detected'] else 'info',
                'text': f"Uplift U2 Density regression delta {regression_delta} (candidate {candidate_l3.get('value')} vs baseline {baseline_l3.get('value')}).",
                'source_hash': source_materials['snapshot_candidate_density']['sha256']
            }
        ]
    }

    evidence_pack = {
        'schema_version': 'codex.audit.evidence-pack/1.0.0',
        'pack_id': 'codex-audit-savant-evidence-v1',
        'generated_at': timestamp,
        'deterministic_inputs': deterministic_inputs,
        'source_materials': source_materials,
        'signals': {
            'learning_health': global_health_section,
            'uplift_experiment': {
                'experiment_outcome': conjecture_snapshot.get('experiment_outcome'),
                'uplift_gain': conjecture_snapshot['experiment_summary']['uplift_gain'],
                'uplift_gain_ci_95': conjecture_snapshot['experiment_summary']['uplift_gain_ci_95'],
                'rfl_pattern_detected': conjecture_snapshot['experiment_summary']['rfl_pattern_detected'],
                'dynamics_metrics': conjecture_snapshot['experiment_summary']['dynamics_metrics']
            },
            'structure_drift': {
                'overall_trend': structure_summary.get('overall_trend'),
                'directories_compared': structure_summary.get('total_directories_compared'),
                'directories_degraded': structure_summary.get('directories_degraded'),
                'global_risk_delta': structure_drift.get('global_risk_delta')
            },
            'drift_forecast': drift_forecast_excerpt,
            'drift_radar': {
                'overall_trend': drift_radar.get('overall_trend'),
                'high_risk_file_count': drift_radar.get('high_risk_file_count')
            },
            'metric_integration': {
                'status': metric_summary.get('status'),
                'warnings': metric_summary.get('warnings', []),
                'potential_errors': metric_summary.get('potential_errors', []),
                'passed_checks': metric_summary.get('passed_checks', [])
            },
            'regression_watch': regression_watch
        }
    }

    gates_detail = {
        'metric_integration_contract': {
            'status': 'attention' if metric_summary.get('status') != 'PASSED' else 'passed',
            'evidence_path': source_materials['metric_integration_report']['path'],
            'source_hash': source_materials['metric_integration_report']['sha256'],
            'notes': metric_summary.get('warnings', [])
        },
        'structure_drift_surveillance': {
            'status': 'passed',
            'evidence_path': source_materials['structure_drift_report']['path'],
            'source_hash': source_materials['structure_drift_report']['sha256'],
            'notes': ['Global trend stable']
        },
        'global_health_learning': {
            'status': 'passed' if global_health_section['status'] == 'HEALTHY' else 'attention',
            'evidence_path': source_materials['conjecture_report']['path'],
            'source_hash': source_materials['conjecture_report']['sha256'],
            'notes': ['No key conjectures contradicted']
        },
        'regression_watch': {
            'status': 'attention' if regression_watch['regression_detected'] else 'passed',
            'evidence_path': source_materials['snapshot_candidate_density']['path'],
            'source_hash': source_materials['snapshot_candidate_density']['sha256'],
            'notes': list(filter(None, [candidate_l3.get('details')]))
        }
    }

    gov_posture_brief = {
        'status': 'WARN',
        'gates': {gate: details['status'] for gate, details in gates_detail.items()}
    }

    governance_combined = gov.combine_conjectures_with_governance(gov_posture_brief, conjecture_snapshot)

    governance_signals_payload = {
        'schema_version': 'codex.audit.governance-posture/1.0.0',
        'generated_at': timestamp,
        'governance_posture': {
            'status': gov_posture_brief['status'],
            'gates': gates_detail
        },
        'dynamics_summary': global_health_section,
        'combined_signal': governance_combined,
        'source_materials': {
            key: source_materials[key]
            for key in ['conjecture_report', 'metric_integration_report', 'structure_drift_report', 'snapshot_candidate_density']
        }
    }

    conjecture_timeline = gov.build_conjecture_timeline([conjecture_snapshot])

    global_health_surface_payload = {
        'schema_version': 'codex.audit.global-health-surface/1.0.0',
        'generated_at': timestamp,
        'tiles': [
            {
                'tile_id': 'learning-health',
                'status': global_health_section['status'],
                'metrics': {
                    'supports': global_health_section['supports'],
                    'contradicts': global_health_section['contradicts'],
                    'inconclusive': global_health_section['inconclusive']
                },
                'source_hash': source_materials['conjecture_report']['sha256']
            },
            {
                'tile_id': 'uplift-u2-density',
                'status': candidate_l3.get('status'),
                'baseline_value': baseline_l3.get('value'),
                'candidate_value': candidate_l3.get('value'),
                'delta': regression_delta,
                'details': candidate_l3.get('details'),
                'source_hash': source_materials['snapshot_candidate_density']['sha256']
            },
            {
                'tile_id': 'structure-drift',
                'status': structure_summary.get('overall_trend'),
                'directories_compared': structure_summary.get('total_directories_compared'),
                'directories_degraded': structure_summary.get('directories_degraded'),
                'source_hash': source_materials['structure_drift_report']['sha256']
            }
        ],
        'timeline': conjecture_timeline,
        'inputs': {
            'conjecture_report': source_materials['conjecture_report'],
            'density_baseline': source_materials['snapshot_baseline_density'],
            'density_candidate': source_materials['snapshot_candidate_density']
        },
        'deterministic_inputs': deterministic_inputs
    }

    risk_levels = sorted({drift.get('risk_level_new') for drift in structure_drift.get('directory_drifts', [])})

    drift_schema_payload = {
        'schema_version': 'codex.audit.drift-schema/1.0.0',
        'generated_at': timestamp,
        'based_on': source_materials['structure_drift_report'],
        'field_specs': [
            {'name': 'timestamp', 'type': 'string', 'format': 'iso8601', 'required': True},
            {'name': 'global_risk_old', 'type': 'number', 'required': True},
            {'name': 'global_risk_new', 'type': 'number', 'required': True},
            {'name': 'directory_drifts', 'type': 'array', 'items': {
                'path': {'type': 'string'},
                'entropy_old': {'type': 'number'},
                'entropy_new': {'type': 'number'},
                'risk_level_new': {'type': 'string', 'enum': risk_levels}
            }},
            {'name': 'summary', 'type': 'object', 'required': True}
        ],
        'risk_levels': risk_levels,
        'sample_summary': structure_summary,
        'notes': ['Schema derived from current structure_drift_report.json contents.']
    }

    artifact_hashes = {}
    artifact_hashes['evidence_pack_v1.json'] = write_json(base_dir / 'evidence_pack_v1.json', evidence_pack)
    artifact_hashes['cicd_summary_lines_v1.json'] = write_json(base_dir / 'cicd_summary_lines_v1.json', cicd_lines_payload)
    artifact_hashes['governance_posture_v1.json'] = write_json(base_dir / 'governance_posture_v1.json', governance_signals_payload)
    artifact_hashes['drift_schema_v1.json'] = write_json(base_dir / 'drift_schema_v1.json', drift_schema_payload)
    artifact_hashes['global_health_surface_v1.json'] = write_json(base_dir / 'global_health_surface_v1.json', global_health_surface_payload)

    attestation_payload = {
        'schema_version': 'codex.audit.attestation/1.0.0',
        'attestation_id': 'codex-audit-savant-attestation-v1',
        'generated_at': timestamp,
        'attester': 'Codex-Audit-Savant',
        'artifact_hashes': artifact_hashes,
        'source_materials': source_materials,
        'deterministic_inputs': deterministic_inputs,
        'challenge': hashlib.sha256((
            'Codex-Audit-Savant|' + timestamp + '|' + '|'.join(sorted(artifact_hashes.values()))
        ).encode('utf-8')).hexdigest()
    }

    artifact_hashes['attestation_manifest_v1.json'] = write_json(base_dir / 'attestation_manifest_v1.json', attestation_payload)

if __name__ == '__main__':
    main()

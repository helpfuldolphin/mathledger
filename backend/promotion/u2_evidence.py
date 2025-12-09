# PHASE II — U2 UPLIFT EXPERIMENT
"""
Operationalizes the U2 Evidence Dossier Specification.
"""
import dataclasses
import datetime
import json
import os
from typing import Dict, List, Any, Optional

ARTIFACT_IDS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
CORE_ARTIFACT_IDS = ["A1", "A2", "A3", "A5", "A7", "A8"] # A4 and A6 are SUPPORTING
REQUIRED_ENVIRONMENTS = ["ENV_A", "ENV_B", "ENV_C", "ENV_D"]

@dataclasses.dataclass(frozen=True)
class U2Artifact:
    id: str
    description: str
    location_template: str
    versioning_scheme: str
    is_multi_instance: bool = False
    found: bool = False
    paths: List[str] = dataclasses.field(default_factory=list)
    @property
    def path(self) -> Optional[str]:
        if not self.is_multi_instance and self.paths:
            return self.paths[0]
        return None

@dataclasses.dataclass
class U2Dossier:
    run_id: str
    creation_timestamp: str
    artifacts: Dict[str, U2Artifact]
    dossier_status: str = "PENDING"

def _get_artifact_definitions() -> Dict[str, U2Artifact]:
    return {
        "A1": U2Artifact(id="A1", description="Preregistration", location_template="docs/prereg/PREREG_UPLIFT_U2.yaml", versioning_scheme="Git Hash"),
        "A2": U2Artifact(id="A2", description="Preregistration Seal", location_template="docs/prereg/PREREG_UPLIFT_U2.yaml.sig", versioning_scheme="Timestamp"),
        "A3": U2Artifact(id="A3", description="Environment Manifests", location_template="artifacts/u2/{ENV_ID}/manifest.json", versioning_scheme="Run ID", is_multi_instance=True),
        "A4": U2Artifact(id="A4", description="Raw Telemetry", location_template="artifacts/u2/{ENV_ID}/telemetry_{RUN_ID}.jsonl", versioning_scheme="Run ID", is_multi_instance=True),
        "A5": U2Artifact(id="A5", description="Statistical Summary", location_template="artifacts/u2/analysis/statistical_summary_{RUN_ID}.json", versioning_scheme="Run ID"),
        "A6": U2Artifact(id="A6", description="Uplift Curve Plots", location_template="artifacts/u2/analysis/uplift_curve_{ENV_ID}_{RUN_ID}.png", versioning_scheme="Run ID", is_multi_instance=True),
        "A7": U2Artifact(id="A7", description="Gate Compliance Log", location_template="ops/logs/u2_compliance.jsonl", versioning_scheme="Timestamp"),
        "A8": U2Artifact(id="A8", description="Governance Verifier Report", location_template="artifacts/u2/governance_verifier_report.json", versioning_scheme="Run ID"),
    }

def assemble_dossier(run_id: str, root_path: str = ".", env_ids: List[str] = None) -> U2Dossier:
    if env_ids is None: env_ids = REQUIRED_ENVIRONMENTS
    dossier = U2Dossier(run_id=run_id, creation_timestamp=datetime.datetime.utcnow().isoformat(), artifacts=_get_artifact_definitions())
    all_core_found = True
    for artifact in dossier.artifacts.values():
        object.__setattr__(artifact, "paths", [])
        if artifact.is_multi_instance:
            found_any = False
            for env_id in env_ids:
                search_path = os.path.join(root_path, artifact.location_template.format(ENV_ID=env_id, RUN_ID=run_id))
                if os.path.exists(search_path):
                    artifact.paths.append(search_path)
                    found_any = True
            object.__setattr__(artifact, "found", found_any)
        else:
            search_path = os.path.join(root_path, artifact.location_template.format(RUN_ID=run_id))
            if os.path.exists(search_path):
                object.__setattr__(artifact, "found", True)
                artifact.paths.append(search_path)
        if not artifact.found and artifact.id in CORE_ARTIFACT_IDS:
            all_core_found = False
    dossier.dossier_status = "COMPLETE" if all_core_found else "INCOMPLETE"
    return dossier
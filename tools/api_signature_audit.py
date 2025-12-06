#!/usr/bin/env python3
"""
CLAUDE I — Structural Grammarian
API Signature & Field Contract Verification
"""

import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set


class APISignatureAuditor:
    def __init__(self, root_path: Path):
        self.root = root_path
        self.schemas_file = root_path / "backend" / "api" / "schemas.py"
        self.violations: List[Dict[str, Any]] = []
        self.models: Dict[str, Dict[str, Any]] = {}

    def parse_pydantic_models(self) -> Dict[str, Dict[str, Any]]:
        """Parse Pydantic models from schemas.py using AST"""
        if not self.schemas_file.exists():
            return {}

        with open(self.schemas_file, "r") as f:
            source = f.read()

        tree = ast.parse(source)
        models = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it inherits from BaseModel
                if any(
                    isinstance(base, ast.Name) and base.id == "BaseModel"
                    for base in node.bases
                ):
                    model_name = node.name
                    fields = {}
                    docstring = ast.get_docstring(node)

                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(
                            item.target, ast.Name
                        ):
                            field_name = item.target.id
                            field_type = ast.unparse(item.annotation)

                            # Check if field has default value
                            has_default = item.value is not None
                            is_optional = "Optional" in field_type or "None" in field_type

                            fields[field_name] = {
                                "type": field_type,
                                "optional": is_optional,
                                "has_default": has_default,
                            }

                    models[model_name] = {
                        "fields": fields,
                        "docstring": docstring,
                        "field_count": len(fields),
                    }

        return models

    def verify_field_contracts(self) -> List[Dict[str, Any]]:
        """Verify field contracts and type consistency"""
        issues = []

        for model_name, model_info in self.models.items():
            fields = model_info["fields"]

            # Check for common issues
            for field_name, field_info in fields.items():
                field_type = field_info["type"]

                # Issue 1: Dict[str, Any] is too permissive
                if "Dict[str, Any]" in field_type or "dict" in field_type.lower():
                    issues.append({
                        "model": model_name,
                        "field": field_name,
                        "type": "permissive_typing",
                        "severity": "warning",
                        "message": f"Field uses permissive Dict[str, Any] - consider stronger typing",
                    })

                # Issue 2: Optional fields without defaults
                if field_info["optional"] and not field_info["has_default"]:
                    # This is actually fine in Pydantic 2.x, but note it
                    pass

                # Issue 3: Non-optional fields that should be optional
                common_optional_fields = {
                    "description", "method", "duration_ms", "error",
                    "parent_id", "derivation_rule", "derivation_depth"
                }
                if field_name in common_optional_fields and not field_info["optional"]:
                    issues.append({
                        "model": model_name,
                        "field": field_name,
                        "type": "missing_optional",
                        "severity": "warning",
                        "message": f"Field '{field_name}' is typically optional but not marked as such",
                    })

        return issues

    def verify_model_completeness(self) -> List[Dict[str, Any]]:
        """Verify models have proper documentation and structure"""
        issues = []

        for model_name, model_info in self.models.items():
            # Check for docstring
            if not model_info["docstring"]:
                issues.append({
                    "model": model_name,
                    "type": "missing_docstring",
                    "severity": "info",
                    "message": "Model lacks docstring documentation",
                })

            # Check for empty models
            if model_info["field_count"] == 0:
                issues.append({
                    "model": model_name,
                    "type": "empty_model",
                    "severity": "error",
                    "message": "Model has no fields defined",
                })

            # Check for created_at timestamp field (common pattern)
            fields = model_info["fields"]
            if "Base" in model_name or "Response" in model_name:
                if "created_at" not in fields and "timestamp" not in fields:
                    # This is just informational, not all models need timestamps
                    pass

        return issues

    def generate_api_surface_map(self) -> Dict[str, Any]:
        """Generate a map of the API surface area"""
        surface_map = {
            "total_models": len(self.models),
            "models": {},
        }

        # Categorize models
        categories = {
            "base": [],
            "response": [],
            "metrics": [],
            "domain": [],
        }

        for model_name in self.models.keys():
            if "Base" in model_name:
                categories["base"].append(model_name)
            elif "Response" in model_name:
                categories["response"].append(model_name)
            elif "Metrics" in model_name:
                categories["metrics"].append(model_name)
            else:
                categories["domain"].append(model_name)

            # Add field summary
            surface_map["models"][model_name] = {
                "field_count": self.models[model_name]["field_count"],
                "fields": list(self.models[model_name]["fields"].keys()),
            }

        surface_map["categories"] = categories

        return surface_map

    def run_audit(self) -> Dict[str, Any]:
        """Run complete API signature audit"""
        print("[Structural Grammarian] Analyzing API signatures...")

        # Parse models
        self.models = self.parse_pydantic_models()

        if not self.models:
            return {
                "error": "No Pydantic models found",
                "models_found": 0,
            }

        # Run verification checks
        field_issues = self.verify_field_contracts()
        completeness_issues = self.verify_model_completeness()

        # Generate surface map
        surface_map = self.generate_api_surface_map()

        # Categorize issues by severity
        errors = [i for i in (field_issues + completeness_issues) if i.get("severity") == "error"]
        warnings = [i for i in (field_issues + completeness_issues) if i.get("severity") == "warning"]
        info = [i for i in (field_issues + completeness_issues) if i.get("severity") == "info"]

        return {
            "summary": {
                "models_analyzed": len(self.models),
                "total_issues": len(errors) + len(warnings) + len(info),
                "errors": len(errors),
                "warnings": len(warnings),
                "info": len(info),
            },
            "issues": {
                "errors": errors,
                "warnings": warnings,
                "info": info,
            },
            "surface_map": surface_map,
        }

    def format_report(self, report: Dict[str, Any]) -> str:
        """Format audit report for output"""
        lines = []
        lines.append("=" * 80)
        lines.append("CLAUDE I — Structural Grammarian")
        lines.append("API Signature & Field Contract Verification")
        lines.append("=" * 80)
        lines.append("")

        if "error" in report:
            lines.append(f"ERROR: {report['error']}")
            return "\n".join(lines)

        summary = report["summary"]
        lines.append(f"Models Analyzed: {summary['models_analyzed']}")
        lines.append(f"Total Issues: {summary['total_issues']}")
        lines.append(f"  Errors: {summary['errors']}")
        lines.append(f"  Warnings: {summary['warnings']}")
        lines.append(f"  Info: {summary['info']}")
        lines.append("")

        # Show surface map categories
        surface = report["surface_map"]
        categories = surface["categories"]
        lines.append("API Surface Area:")
        lines.append(f"  Base Models: {len(categories['base'])}")
        lines.append(f"  Response Models: {len(categories['response'])}")
        lines.append(f"  Metrics Models: {len(categories['metrics'])}")
        lines.append(f"  Domain Models: {len(categories['domain'])}")
        lines.append("")

        # Show issues
        issues = report["issues"]

        if issues["errors"]:
            lines.append("=" * 80)
            lines.append("ERRORS")
            lines.append("=" * 80)
            for err in issues["errors"]:
                lines.append(f"[{err['model']}]")
                lines.append(f"  Type: {err['type']}")
                lines.append(f"  {err['message']}")
                lines.append("")

        if issues["warnings"]:
            lines.append("=" * 80)
            lines.append("WARNINGS")
            lines.append("=" * 80)
            for warn in issues["warnings"]:
                if "field" in warn:
                    lines.append(f"[{warn['model']}.{warn['field']}]")
                else:
                    lines.append(f"[{warn['model']}]")
                lines.append(f"  Type: {warn['type']}")
                lines.append(f"  {warn['message']}")
                lines.append("")

        lines.append("=" * 80)
        if summary["errors"] == 0:
            lines.append("[PASS] API Signature Conformance OK")
        else:
            lines.append(f"[FAIL] {summary['errors']} error(s) detected")
        lines.append("=" * 80)

        return "\n".join(lines)


def main():
    root = Path(__file__).parent.parent
    auditor = APISignatureAuditor(root)

    report = auditor.run_audit()

    # Print formatted report
    print(auditor.format_report(report))

    # Save detailed report
    report_path = root / "artifacts" / "api_signature_report.json"
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved: {report_path.relative_to(root)}")

    # Exit with non-zero if errors found
    sys.exit(1 if report.get("summary", {}).get("errors", 0) > 0 else 0)


if __name__ == "__main__":
    main()

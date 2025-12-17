"""Advisory verifier for ledger_guard_summary.json artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional


def _compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_manifest_path(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).as_posix()


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _discover_summary_path(pack_root: Path) -> Optional[Path]:
    candidates = [
        pack_root / "governance" / "ledger_guard_summary.json",
        pack_root / "compliance" / "ledger_guard_summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _discover_raw_output_path(pack_root: Path) -> Optional[Path]:
    candidates = [
        pack_root / "governance" / "ledger_guard_raw.json",
        pack_root / "governance" / "ledger_guard_output.json",
        pack_root / "governance" / "ledger_guard_check_result.json",
        pack_root / "governance" / "ledger_guard_result.json",
        pack_root / "governance" / "ledger_guard.json",
        pack_root / "compliance" / "ledger_guard_raw.json",
        pack_root / "compliance" / "ledger_guard_output.json",
        pack_root / "compliance" / "ledger_guard_check_result.json",
        pack_root / "compliance" / "ledger_guard_result.json",
        pack_root / "compliance" / "ledger_guard.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_files_entry(manifest: Dict[str, Any], rel_path: str) -> Optional[Dict[str, Any]]:
    wanted = _normalize_manifest_path(rel_path)
    for entry in manifest.get("files") or []:
        if not isinstance(entry, dict):
            continue
        path_value = entry.get("path")
        if isinstance(path_value, str) and _normalize_manifest_path(path_value) == wanted:
            return entry
    return None


def verify_ledger_guard_summary(
    *,
    pack_dir: Optional[Path],
    manifest_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    raw_output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify ledger guard summary integrity against the manifest."""
    resolved_pack_dir = pack_dir.resolve() if pack_dir else None
    resolved_manifest_path = manifest_path.resolve() if manifest_path else None
    if resolved_manifest_path is None:
        if resolved_pack_dir is None:
            return {
                "pack_dir": None,
                "manifest_path": None,
                "summary_path": None,
                "raw_output_path": None,
                "file_hash_match": None,
                "schema_version_match": None,
                "violation_count_match": None,
                "status_light_match": None,
                "summary_internal_counts_match": None,
                "all_checks_passed": False,
                "errors": ["pack_dir or manifest_path must be provided"],
                "warnings": [],
            }
        resolved_manifest_path = resolved_pack_dir / "manifest.json"

    pack_root = resolved_pack_dir if resolved_pack_dir else resolved_manifest_path.parent

    result: Dict[str, Any] = {
        "pack_dir": str(pack_root),
        "manifest_path": str(resolved_manifest_path),
        "summary_path": None,
        "raw_output_path": None,
        "file_hash_match": None,
        "schema_version_match": None,
        "violation_count_match": None,
        "status_light_match": None,
        "summary_internal_counts_match": None,
        "all_checks_passed": False,
        "errors": [],
        "warnings": [],
    }

    if not resolved_manifest_path.exists():
        result["errors"].append(f"Manifest not found: {resolved_manifest_path}")
        return result

    try:
        manifest = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result["errors"].append(f"Failed to parse manifest: {exc}")
        return result

    ledger_meta = (
        (manifest.get("governance") or {})
        .get("schema_versioned", {})
        .get("ledger_guard_summary")
    )
    if not isinstance(ledger_meta, dict):
        ledger_meta = None

    expected_sha256 = ledger_meta.get("sha256") if ledger_meta else None
    meta_schema_version = ledger_meta.get("schema_version") if ledger_meta else None

    if summary_path is None:
        summary_ref = ledger_meta.get("path") if ledger_meta else None
        if isinstance(summary_ref, str) and summary_ref.strip():
            normalized = _normalize_manifest_path(summary_ref)
            summary_path = (pack_root / Path(normalized)).resolve()
        else:
            summary_path = _discover_summary_path(pack_root)
            if summary_path is None:
                result["errors"].append("Ledger guard summary not found in pack")
                return result
    else:
        summary_path = Path(summary_path).resolve()

    result["summary_path"] = str(summary_path)

    if not summary_path.exists():
        result["errors"].append(f"Summary file not found: {summary_path}")
        return result

    rel_summary_path = _normalize_manifest_path(
        str(summary_path.relative_to(pack_root))
        if summary_path.is_relative_to(pack_root)
        else str(summary_path)
    )

    if expected_sha256 is None:
        files_entry = _find_files_entry(manifest, rel_summary_path)
        if files_entry is not None:
            expected_sha256 = files_entry.get("sha256")

    if expected_sha256 is not None:
        file_hash = _compute_file_hash(summary_path)
        result["file_hash_match"] = file_hash == expected_sha256
    else:
        result["warnings"].append("Unable to resolve expected sha256 for summary file")

    try:
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result["errors"].append(f"Failed to parse ledger guard summary JSON: {exc}")
        return result

    if not isinstance(summary_payload, dict):
        result["errors"].append("Ledger guard summary must be a JSON object")
        return result

    summary_schema_version = summary_payload.get("schema_version")
    if meta_schema_version is not None and summary_schema_version is not None:
        result["schema_version_match"] = meta_schema_version == summary_schema_version

    status_light = summary_payload.get("status_light")
    violation_counts = summary_payload.get("violation_counts")
    if violation_counts is None:
        violation_counts = summary_payload.get("violation_count")

    summary_violation_counts = _coerce_int(violation_counts)
    summary_violation_count = _coerce_int(summary_payload.get("violation_count"))
    if summary_violation_counts is not None and summary_violation_count is not None:
        result["summary_internal_counts_match"] = (
            summary_violation_counts == summary_violation_count
        )

    raw_payload: Optional[Dict[str, Any]] = None
    if raw_output_path is None:
        raw_output_path = _discover_raw_output_path(pack_root)
    if raw_output_path is not None:
        raw_output_path = Path(raw_output_path).resolve()
        result["raw_output_path"] = str(raw_output_path)

        if raw_output_path.exists():
            try:
                raw_payload = json.loads(raw_output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                result["warnings"].append(f"Failed to parse raw guard output JSON: {exc}")
                result["violation_count_match"] = False

            if isinstance(raw_payload, dict):
                violations = raw_payload.get("violations")
                if isinstance(violations, list):
                    recomputed_count = len(violations)
                    if summary_violation_counts is None:
                        result["warnings"].append(
                            "Summary missing violation_counts; cannot compare to recomputed count"
                        )
                        result["violation_count_match"] = False
                    else:
                        result["violation_count_match"] = (
                            recomputed_count == summary_violation_counts
                        )
                else:
                    result["warnings"].append(
                        "Raw guard output missing violations list; skipping recompute"
                    )
                    result["violation_count_match"] = False
            else:
                result["warnings"].append(
                    "Raw guard output payload is not an object; skipping recompute"
                )
                result["violation_count_match"] = False
        else:
            result["warnings"].append(f"Raw guard output not found: {raw_output_path}")

    if raw_output_path is not None and isinstance(raw_payload, dict):
        try:
            from backend.health.ledger_guard_tile import build_ledger_guard_tile

            recomputed_tile = build_ledger_guard_tile(raw_payload)
            result["status_light_match"] = recomputed_tile.get("status_light") == status_light
        except Exception as exc:  # pragma: no cover - advisory only
            result["warnings"].append(f"Failed to recompute status_light from raw output: {exc}")
            result["status_light_match"] = False

    checks = [
        check
        for check in (
            result["file_hash_match"],
            result["schema_version_match"],
            result["summary_internal_counts_match"],
            result["violation_count_match"],
            result["status_light_match"],
        )
        if check is not None
    ]
    result["all_checks_passed"] = bool(checks) and all(checks) and not result["errors"]
    if not checks and not result["errors"]:
        result["all_checks_passed"] = True
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Advisory ledger guard summary verifier (non-gating)."
    )
    parser.add_argument(
        "--pack-dir",
        type=str,
        help="Evidence pack directory containing manifest.json and summary.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Path to manifest.json (defaults to <pack-dir>/manifest.json).",
    )
    parser.add_argument(
        "--summary",
        type=str,
        help="Explicit path to ledger_guard_summary.json (optional).",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        help="Optional path to raw ledger guard output JSON (violations list).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path for deterministic JSON report output.",
    )
    return parser


def _build_output_report(result: Dict[str, Any]) -> Dict[str, Any]:
    sha256_ok = result.get("file_hash_match") is True
    recompute_ok = result.get("violation_count_match")

    reason_codes: set[str] = set()
    summary_path = result.get("summary_path")
    if not summary_path or not Path(summary_path).exists():
        reason_codes.add("MISSING_SUMMARY")
    if result.get("file_hash_match") is False:
        reason_codes.add("SHA256_MISMATCH")
    if recompute_ok is False or result.get("status_light_match") is False:
        reason_codes.add("RECOMPUTE_FAILED")

    raw_output_path = result.get("raw_output_path")
    if not raw_output_path or not Path(raw_output_path).exists():
        reason_codes.add("RAW_OUTPUT_MISSING")

    ordered_reason_codes = (
        "MISSING_SUMMARY",
        "SHA256_MISMATCH",
        "RECOMPUTE_FAILED",
        "RAW_OUTPUT_MISSING",
    )
    notes = sorted(
        reason_codes,
        key=lambda code: (
            ordered_reason_codes.index(code)
            if code in ordered_reason_codes
            else len(ordered_reason_codes),
            code,
        ),
    )

    return {
        "pack_dir": result.get("pack_dir"),
        "summary_path": summary_path,
        "sha256_ok": bool(sha256_ok),
        "recompute_ok": recompute_ok,
        "notes": notes,
    }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    pack_dir = Path(args.pack_dir) if args.pack_dir else None
    manifest_path = Path(args.manifest) if args.manifest else None
    summary_path = Path(args.summary) if args.summary else None
    raw_output_path = Path(args.raw_output) if args.raw_output else None
    output_path = Path(args.output) if args.output else None

    result = verify_ledger_guard_summary(
        pack_dir=pack_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        raw_output_path=raw_output_path,
    )

    report = _build_output_report(result)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if result["errors"]:
        print("Verification errors:")
        for err in result["errors"]:
            print(f"  - {err}")
        return 1

    print(f"Manifest path: {result['manifest_path']}")
    print(f"Summary path:  {result['summary_path']}")
    if result.get("raw_output_path"):
        print(f"Raw output:    {result['raw_output_path']}")

    if result["file_hash_match"] is not None:
        print(f"File hash matches manifest: {result['file_hash_match']}")

    if result["schema_version_match"] is not None:
        print(f"Schema version matches manifest metadata: {result['schema_version_match']}")

    if result["summary_internal_counts_match"] is not None:
        print(
            "Summary violation_count matches violation_counts: "
            f"{result['summary_internal_counts_match']}"
        )

    if result["violation_count_match"] is not None:
        print(f"Violation count matches raw output: {result['violation_count_match']}")

    if result["status_light_match"] is not None:
        print(f"Status light matches raw output: {result['status_light_match']}")

    if result["warnings"]:
        print("Warnings:")
        for warn in result["warnings"]:
            print(f"  - {warn}")

    recompute_ok = report["recompute_ok"]
    passed = bool(report["sha256_ok"]) and (recompute_ok is True or recompute_ok is None)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

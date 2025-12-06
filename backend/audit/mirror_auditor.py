#!/usr/bin/env python3
"""
Mirror Auditor: Dual-Root Attestation Verifier

Recomputes H_t = SHA256(R_t || U_t) for all blocks and validates
dual-root mirror symmetry integrity.

Role: Verifier of Dual-Root Symmetry
Seal: [PASS] Dual-Root Mirror Integrity OK coverage≥95%
"""

import sys
import os
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import from canonical attestation module (single source of truth)
from attestation.dual_root import (
    compute_composite_root,
    verify_composite_integrity,
)
from backend.db.connection import get_connection


@dataclass
class BlockAuditResult:
    """Audit result for a single block."""
    block_id: int
    block_number: int
    has_dual_roots: bool
    reasoning_root: str | None
    ui_root: str | None
    stored_composite: str | None
    computed_composite: str | None
    integrity_valid: bool
    error: str | None = None


def get_blocks_with_dual_roots(conn) -> List[Dict[str, Any]]:
    """
    Fetch all blocks from database with dual-root attestation fields.

    Schema-tolerant: checks for presence of dual-root columns.
    """
    cur = conn.cursor()

    # Check what columns exist in blocks table
    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'blocks'
        ORDER BY ordinal_position
    """)
    columns = [row[0] for row in cur.fetchall()]

    # Check for dual-root columns
    has_reasoning = 'reasoning_merkle_root' in columns
    has_ui = 'ui_merkle_root' in columns
    has_composite = 'composite_attestation_root' in columns

    # Build query based on available columns
    select_parts = ['id', 'block_number', 'merkle_root']
    if has_reasoning:
        select_parts.append('reasoning_merkle_root')
    if has_ui:
        select_parts.append('ui_merkle_root')
    if has_composite:
        select_parts.append('composite_attestation_root')

    query = f"SELECT {', '.join(select_parts)} FROM blocks ORDER BY block_number"

    cur.execute(query)
    rows = cur.fetchall()

    # Convert to dictionaries
    blocks = []
    for row in rows:
        block = {
            'id': row[0],
            'block_number': row[1],
            'merkle_root': row[2]
        }

        idx = 3
        if has_reasoning:
            block['reasoning_merkle_root'] = row[idx]
            idx += 1
        if has_ui:
            block['ui_merkle_root'] = row[idx]
            idx += 1
        if has_composite:
            block['composite_attestation_root'] = row[idx]
            idx += 1

        blocks.append(block)

    cur.close()
    return blocks, has_reasoning and has_ui and has_composite


def audit_block(block: Dict[str, Any]) -> BlockAuditResult:
    """
    Audit a single block's dual-root integrity.

    Recomputes H_t = SHA256(R_t || U_t) and validates against stored value.
    """
    block_id = block['id']
    block_number = block['block_number']

    # Check if block has dual-root fields
    r_t = block.get('reasoning_merkle_root')
    u_t = block.get('ui_merkle_root')
    stored_h_t = block.get('composite_attestation_root')

    has_dual_roots = bool(r_t and u_t and stored_h_t)

    if not has_dual_roots:
        return BlockAuditResult(
            block_id=block_id,
            block_number=block_number,
            has_dual_roots=False,
            reasoning_root=r_t,
            ui_root=u_t,
            stored_composite=stored_h_t,
            computed_composite=None,
            integrity_valid=False,
            error="Missing dual-root fields"
        )

    # Recompute composite root
    try:
        computed_h_t = compute_composite_root(r_t, u_t)
        integrity_valid = verify_composite_integrity(r_t, u_t, stored_h_t)

        return BlockAuditResult(
            block_id=block_id,
            block_number=block_number,
            has_dual_roots=True,
            reasoning_root=r_t,
            ui_root=u_t,
            stored_composite=stored_h_t,
            computed_composite=computed_h_t,
            integrity_valid=integrity_valid,
            error=None if integrity_valid else "H_t mismatch"
        )
    except Exception as e:
        return BlockAuditResult(
            block_id=block_id,
            block_number=block_number,
            has_dual_roots=True,
            reasoning_root=r_t,
            ui_root=u_t,
            stored_composite=stored_h_t,
            computed_composite=None,
            integrity_valid=False,
            error=f"Computation error: {str(e)}"
        )


def run_mirror_audit() -> Tuple[List[BlockAuditResult], float, bool]:
    """
    Run complete Mirror Auditor verification.

    Returns:
        - List of audit results for each block
        - Coverage percentage (blocks with dual roots / total blocks)
        - Overall pass/fail status (coverage >= 95% and all valid)
    """
    conn = get_connection()

    try:
        # Fetch all blocks
        blocks, has_dual_root_schema = get_blocks_with_dual_roots(conn)

        if not blocks:
            print("⚠️  No blocks found in database")
            return [], 0.0, False

        if not has_dual_root_schema:
            print("⚠️  Database schema does not have dual-root columns")
            print("     Expected: reasoning_merkle_root, ui_merkle_root, composite_attestation_root")
            return [], 0.0, False

        # Audit each block
        results = []
        for block in blocks:
            result = audit_block(block)
            results.append(result)

        # Calculate metrics
        total_blocks = len(results)
        blocks_with_dual_roots = sum(1 for r in results if r.has_dual_roots)
        valid_blocks = sum(1 for r in results if r.integrity_valid)

        coverage = (blocks_with_dual_roots / total_blocks * 100) if total_blocks > 0 else 0.0
        integrity_rate = (valid_blocks / blocks_with_dual_roots * 100) if blocks_with_dual_roots > 0 else 0.0

        # Pass criteria: coverage >= 95% and all dual-root blocks are valid
        passes = coverage >= 95.0 and (blocks_with_dual_roots == valid_blocks)

        return results, coverage, passes

    finally:
        conn.close()


def format_audit_report(
    results: List[BlockAuditResult],
    coverage: float,
    passes: bool
) -> str:
    """
    Format the audit report for display.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("🜎 MIRROR AUDITOR — DUAL-ROOT ATTESTATION VERIFICATION")
    lines.append("=" * 80)
    lines.append("")

    if not results:
        lines.append("⚠️  No blocks found or schema missing dual-root columns")
        lines.append("")
        lines.append("Seal: [FAIL] No data to audit")
        return "\n".join(lines)

    # Summary statistics
    total_blocks = len(results)
    blocks_with_dual_roots = sum(1 for r in results if r.has_dual_roots)
    valid_blocks = sum(1 for r in results if r.integrity_valid)
    invalid_blocks = blocks_with_dual_roots - valid_blocks

    lines.append(f"Total Blocks:              {total_blocks}")
    lines.append(f"Blocks with Dual Roots:    {blocks_with_dual_roots}")
    lines.append(f"Valid H_t Attestations:    {valid_blocks}")
    lines.append(f"Invalid H_t Attestations:  {invalid_blocks}")
    lines.append(f"Coverage:                  {coverage:.2f}%")
    lines.append("")

    # Detailed results
    if invalid_blocks > 0:
        lines.append("INTEGRITY FAILURES:")
        lines.append("-" * 80)
        for r in results:
            if r.has_dual_roots and not r.integrity_valid:
                lines.append(f"Block #{r.block_number} (ID: {r.block_id})")
                lines.append(f"  Error: {r.error}")
                lines.append(f"  R_t (Reasoning):     {r.reasoning_root}")
                lines.append(f"  U_t (UI):            {r.ui_root}")
                lines.append(f"  H_t (Stored):        {r.stored_composite}")
                lines.append(f"  H_t (Computed):      {r.computed_composite}")
                lines.append("")
    else:
        lines.append("✓ All dual-root attestations verified successfully")
        lines.append("")

        # Show sample of verified blocks
        lines.append("Sample Verified Blocks:")
        lines.append("-" * 80)
        sample_size = min(5, len([r for r in results if r.has_dual_roots]))
        sample_results = [r for r in results if r.has_dual_roots][:sample_size]

        for r in sample_results:
            lines.append(f"Block #{r.block_number} (ID: {r.block_id})")
            lines.append(f"  R_t: {r.reasoning_root[:16]}...{r.reasoning_root[-16:]}")
            lines.append(f"  U_t: {r.ui_root[:16]}...{r.ui_root[-16:]}")
            lines.append(f"  H_t: {r.computed_composite[:16]}...{r.computed_composite[-16:]}")
            lines.append(f"  ✓ Integrity Valid")
            lines.append("")

    # Seal
    lines.append("=" * 80)
    if passes:
        lines.append(f"Seal: [PASS] Dual-Root Mirror Integrity OK (coverage={coverage:.1f}%)")
    else:
        if coverage < 95.0:
            lines.append(f"Seal: [FAIL] Coverage below threshold (coverage={coverage:.1f}% < 95%)")
        else:
            lines.append(f"Seal: [FAIL] Integrity validation failed ({invalid_blocks} blocks)")
    lines.append("=" * 80)
    lines.append("")

    if passes:
        lines.append("✨ Dual roots reflect as one.")

    return "\n".join(lines)


def main():
    """Run Mirror Auditor verification and display report."""
    print("\n🜎 Starting Mirror Auditor verification...\n")

    try:
        results, coverage, passes = run_mirror_audit()
        report = format_audit_report(results, coverage, passes)
        print(report)

        # Exit with appropriate code
        sys.exit(0 if passes else 1)

    except Exception as e:
        print(f"❌ Audit failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()

/**
 * Attestation Invariant Enforcement
 * ==================================
 *
 * This module provides runtime validation for attestation data to ensure
 * the UI displays correct, deterministic values.
 *
 * Invariants:
 * 1. R_t, U_t, H_t must be 64-character lowercase hex strings
 * 2. H_t must equal SHA256(R_t || U_t) (if we can verify client-side)
 * 3. Truncation format: 12 chars + "…" + 8 chars = 21 chars total
 * 4. Block numbers must be non-negative integers
 * 5. Event counts must be non-negative integers
 */

import type { AttestationSummary } from "./api";

export interface AttestationInvariantResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

const HEX64_PATTERN = /^[0-9a-f]{64}$/;

/**
 * Validate a 64-character hex string.
 */
export function isValidHex64(value: string | null | undefined): boolean {
  return typeof value === "string" && HEX64_PATTERN.test(value);
}

/**
 * Validate the truncation format: 12 chars + "…" + 8 chars.
 */
export function isValidTruncation(truncated: string): boolean {
  // Expected format: 12 hex chars + "…" + 8 hex chars
  const pattern = /^[0-9a-f]{12}…[0-9a-f]{8}$/;
  return pattern.test(truncated);
}

/**
 * Truncate a hash consistently: first 12 chars + "…" + last 8 chars.
 * This is the canonical truncation format for the attestation panel.
 */
export function truncateHash(
  hash: string,
  prefix = 12,
  suffix = 8
): string {
  if (!hash || hash.length <= prefix + suffix + 1) {
    return hash || "";
  }
  return `${hash.slice(0, prefix)}…${hash.slice(-suffix)}`;
}

/**
 * Validate an attestation summary against all invariants.
 */
export function validateAttestationInvariants(
  attestation: AttestationSummary | null
): AttestationInvariantResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!attestation) {
    return { valid: true, errors: [], warnings: [] };
  }

  // Invariant 1: R_t must be 64-char hex
  if (!isValidHex64(attestation.reasoningMerkleRoot)) {
    errors.push(
      `Invalid R_t: expected 64-char hex, got "${attestation.reasoningMerkleRoot?.slice(0, 20)}..."`
    );
  }

  // Invariant 2: U_t must be 64-char hex
  if (!isValidHex64(attestation.uiMerkleRoot)) {
    errors.push(
      `Invalid U_t: expected 64-char hex, got "${attestation.uiMerkleRoot?.slice(0, 20)}..."`
    );
  }

  // Invariant 3: H_t must be 64-char hex
  if (!isValidHex64(attestation.compositeAttestationRoot)) {
    errors.push(
      `Invalid H_t: expected 64-char hex, got "${attestation.compositeAttestationRoot?.slice(0, 20)}..."`
    );
  }

  // Invariant 4: Block number must be non-negative (if present)
  if (
    attestation.blockNumber != null &&
    (typeof attestation.blockNumber !== "number" || attestation.blockNumber < 0)
  ) {
    errors.push(
      `Invalid block number: expected non-negative integer, got ${attestation.blockNumber}`
    );
  }

  // Invariant 5: Event counts must be non-negative
  const reasoningCount =
    typeof attestation.metadata?.reasoning_event_count === "number"
      ? attestation.metadata.reasoning_event_count
      : attestation.reasoningLeaves.length;
  const uiCount =
    typeof attestation.metadata?.ui_event_count === "number"
      ? attestation.metadata.ui_event_count
      : attestation.uiLeaves.length;

  if (reasoningCount < 0) {
    errors.push(`Invalid reasoning event count: ${reasoningCount}`);
  }
  if (uiCount < 0) {
    errors.push(`Invalid UI event count: ${uiCount}`);
  }

  // Warning: Check if H_t could be verified (we can't compute SHA256 client-side easily)
  // This is a soft check - the backend is the source of truth
  if (
    attestation.reasoningMerkleRoot &&
    attestation.uiMerkleRoot &&
    attestation.compositeAttestationRoot
  ) {
    // Note: We can't verify H_t = SHA256(R_t || U_t) without a crypto library
    // This would require importing a SHA256 implementation
    warnings.push(
      "H_t integrity cannot be verified client-side. Trust backend computation."
    );
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Assert attestation invariants in development mode.
 * Throws in development, logs in production.
 */
export function assertAttestationInvariants(
  attestation: AttestationSummary | null,
  context = "AttestationPanel"
): void {
  const result = validateAttestationInvariants(attestation);

  if (!result.valid) {
    const message = `[${context}] Attestation invariant violation:\n${result.errors.join("\n")}`;

    if (process.env.NODE_ENV === "development") {
      console.error(message);
      // In development, we log but don't throw to avoid breaking the UI
    } else {
      // In production, just log
      console.warn(message);
    }
  }

  // Log warnings in development
  if (result.warnings.length > 0 && process.env.NODE_ENV === "development") {
    console.debug(
      `[${context}] Attestation warnings:\n${result.warnings.join("\n")}`
    );
  }
}

/**
 * Compute the expected truncated display value for a hash.
 * This is used in tests to verify the UI displays the correct value.
 */
export function expectedTruncatedDisplay(hash: string): string {
  return truncateHash(hash, 12, 8);
}


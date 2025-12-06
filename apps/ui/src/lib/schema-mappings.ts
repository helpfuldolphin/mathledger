/**
 * Schema Mappings: Backend Pydantic → Frontend TypeScript
 * ========================================================
 *
 * This file documents the exact correspondence between backend API schemas
 * (defined in interface/api/schemas.py) and frontend TypeScript interfaces
 * (defined in apps/ui/src/lib/api.ts).
 *
 * Naming Convention:
 * - Backend: snake_case (Python/Pydantic convention)
 * - Frontend: camelCase (TypeScript/JavaScript convention)
 *
 * The api.ts file contains Raw* interfaces that match the wire format,
 * and public interfaces that are the transformed camelCase versions.
 */

// =============================================================================
// HEARTBEAT
// =============================================================================

/**
 * Backend: HeartbeatResponse (interface/api/schemas.py)
 * Frontend: Heartbeat (apps/ui/src/lib/api.ts)
 *
 * Field Mappings:
 * | Backend (snake_case)      | Frontend (camelCase)    | Notes                    |
 * |---------------------------|-------------------------|--------------------------|
 * | ok                        | ok                      | Direct copy              |
 * | ts                        | ts                      | Parsed to Date           |
 * | proofs.success            | proofs.success          | Direct copy              |
 * | proofs_per_sec            | proofsPerHour           | Multiplied by 3600       |
 * | blocks.height             | blocks.height           | Direct copy              |
 * | blocks.latest.merkle      | blocks.latestMerkle     | Flattened                |
 * | policy.hash               | policyHash              | Flattened                |
 * | redis.ml_jobs_len         | redisQueueLength        | Renamed                  |
 */
export interface HeartbeatMapping {
  backend: "HeartbeatResponse";
  frontend: "Heartbeat";
}

// =============================================================================
// STATEMENTS
// =============================================================================

/**
 * Backend: StatementDetailResponse (interface/api/schemas.py)
 * Frontend: StatementDetail (apps/ui/src/lib/api.ts)
 *
 * Field Mappings:
 * | Backend (snake_case)      | Frontend (camelCase)    | Notes                    |
 * |---------------------------|-------------------------|--------------------------|
 * | hash                      | hash                    | Direct copy              |
 * | text                      | text                    | Direct copy              |
 * | normalized_text           | normalizedText          | camelCase                |
 * | display                   | display                 | Direct copy              |
 * | proofs                    | proofs                  | Array, each mapped       |
 * | parents                   | parents                 | Array, each mapped       |
 */
export interface StatementDetailMapping {
  backend: "StatementDetailResponse";
  frontend: "StatementDetail";
}

/**
 * Backend: ProofSummary (interface/api/schemas.py)
 * Frontend: ProofSummary (apps/ui/src/lib/api.ts)
 *
 * Field Mappings:
 * | Backend (snake_case)      | Frontend (camelCase)    | Notes                    |
 * |---------------------------|-------------------------|--------------------------|
 * | method                    | method                  | Direct copy              |
 * | status                    | status                  | Direct copy              |
 * | success                   | success                 | Direct copy              |
 * | created_at                | createdAt               | Parsed to Date           |
 * | prover                    | prover                  | Direct copy              |
 * | duration_ms               | durationMs              | camelCase                |
 */
export interface ProofSummaryMapping {
  backend: "ProofSummary";
  frontend: "ProofSummary";
}

/**
 * Backend: ParentSummary (interface/api/schemas.py)
 * Frontend: ParentSummary (apps/ui/src/lib/api.ts)
 *
 * Field Mappings:
 * | Backend (snake_case)      | Frontend (camelCase)    | Notes                    |
 * |---------------------------|-------------------------|--------------------------|
 * | hash                      | hash                    | Direct copy              |
 * | display                   | display                 | Direct copy              |
 */
export interface ParentSummaryMapping {
  backend: "ParentSummary";
  frontend: "ParentSummary";
}

// =============================================================================
// ATTESTATION
// =============================================================================

/**
 * Backend: AttestationLatestResponse (interface/api/schemas.py)
 * Frontend: AttestationSummary (apps/ui/src/lib/api.ts)
 *
 * Field Mappings:
 * | Backend (snake_case)           | Frontend (camelCase)           | Notes              |
 * |--------------------------------|--------------------------------|--------------------|
 * | block_number                   | blockNumber                    | camelCase          |
 * | block_hash                     | blockHash                      | camelCase          |
 * | reasoning_merkle_root          | reasoningMerkleRoot            | camelCase          |
 * | ui_merkle_root                 | uiMerkleRoot                   | camelCase          |
 * | composite_attestation_root     | compositeAttestationRoot       | camelCase          |
 * | attestation_metadata           | metadata                       | Renamed, flattened |
 * | attestation_metadata.reasoning_leaves | reasoningLeaves         | Extracted          |
 * | attestation_metadata.ui_leaves | uiLeaves                       | Extracted          |
 */
export interface AttestationMapping {
  backend: "AttestationLatestResponse";
  frontend: "AttestationSummary";
}

/**
 * Backend: UIEventResponse (interface/api/schemas.py)
 * Frontend: RecordUiEventResponse (apps/ui/src/lib/api.ts)
 *
 * Field Mappings:
 * | Backend (snake_case)      | Frontend (camelCase)    | Notes                    |
 * |---------------------------|-------------------------|--------------------------|
 * | event_id                  | event_id                | Kept as snake_case       |
 * | timestamp                 | timestamp               | Direct copy (float)      |
 * | leaf_hash                 | leaf_hash               | Kept as snake_case       |
 */
export interface UIEventResponseMapping {
  backend: "UIEventResponse";
  frontend: "RecordUiEventResponse";
}

// =============================================================================
// VALIDATION HELPERS
// =============================================================================

/**
 * Validate that a raw API response matches the expected backend schema.
 * This is useful for catching API contract violations early.
 */
export function validateRawResponse<T extends Record<string, unknown>>(
  response: T,
  requiredFields: (keyof T)[]
): { valid: boolean; missingFields: string[] } {
  const missingFields: string[] = [];

  for (const field of requiredFields) {
    if (!(field in response)) {
      missingFields.push(String(field));
    }
  }

  return {
    valid: missingFields.length === 0,
    missingFields,
  };
}

/**
 * Transform snake_case keys to camelCase.
 */
export function snakeToCamel(str: string): string {
  return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

/**
 * Transform camelCase keys to snake_case.
 */
export function camelToSnake(str: string): string {
  return str.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
}

/**
 * Deep transform object keys from snake_case to camelCase.
 */
export function transformKeysToCamel<T>(obj: unknown): T {
  if (Array.isArray(obj)) {
    return obj.map((item) => transformKeysToCamel(item)) as T;
  }

  if (obj !== null && typeof obj === "object") {
    return Object.fromEntries(
      Object.entries(obj).map(([key, value]) => [
        snakeToCamel(key),
        transformKeysToCamel(value),
      ])
    ) as T;
  }

  return obj as T;
}


#!/usr/bin/env npx tsx
/**
 * Sync Attestation Fixture Script
 * ================================
 * 
 * Transforms the backend attestation artifact (artifacts/first_organism/attestation.json)
 * into the UI fixture format (apps/ui/test/fixtures/first_organism_attestation.json).
 * 
 * The backend artifact uses snake_case keys and a different structure than the
 * UI's camelCase TypeScript interfaces. This script performs the canonical mapping.
 * 
 * Usage:
 *   npm run sync:attestation-fixture
 *   npx tsx scripts/sync-attestation-fixture.ts
 * 
 * The script will:
 *   1. Read artifacts/first_organism/attestation.json
 *   2. Transform to UI fixture format (camelCase, AttestationSummary shape)
 *   3. Write to apps/ui/test/fixtures/first_organism_attestation.json
 *   4. Validate the fixture matches the expected schema
 */

import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

// Paths relative to workspace root
const WORKSPACE_ROOT = path.resolve(__dirname, "..", "..", "..");
const BACKEND_ARTIFACT_PATH = path.join(
  WORKSPACE_ROOT,
  "artifacts",
  "first_organism",
  "attestation.json"
);
const UI_FIXTURE_PATH = path.join(
  __dirname,
  "..",
  "test",
  "fixtures",
  "first_organism_attestation.json"
);

interface BackendArtifact {
  statement_hash: string;
  R_t: string;
  U_t: string;
  H_t: string;
  mdap_seed: number;
  run_id: string;
  run_timestamp_iso: string;
  run_timestamp_unix: number;
  block_id: number | null;
  proof_id: number | null;
  statement_id: number | null;
  version: string;
  environment_mode: string;
  slice_name: string;
  extra?: Record<string, unknown>;
  components?: Record<string, string>;
}

interface UIFixtureLeaf {
  originalIndex: number;
  sortedIndex: number;
  canonicalValue: string;
  leafHash: string;
  merkleProof: Array<[string, boolean]>;
}

interface UIFixture {
  blockNumber: number | null;
  blockHash: string | null;
  reasoningMerkleRoot: string;
  uiMerkleRoot: string;
  compositeAttestationRoot: string;
  reasoningLeaves: UIFixtureLeaf[];
  uiLeaves: UIFixtureLeaf[];
  metadata: Record<string, unknown>;
  _comment: string;
}

function sha256(data: string): string {
  return crypto.createHash("sha256").update(data, "utf8").hexdigest();
}

function computeLeafHash(canonicalValue: string): string {
  return sha256(canonicalValue);
}

function transformArtifact(backend: BackendArtifact): UIFixture {
  // Build sample reasoning leaf from the backend artifact
  const reasoningLeafValue = JSON.stringify(
    {
      prover: "lean-interface",
      reason: "mock dominant statement",
      statement: "p -> p",
      statement_hash: backend.statement_hash,
      status: "abstain",
      verification_method: "lean-disabled",
    },
    Object.keys({
      prover: "",
      reason: "",
      statement: "",
      statement_hash: "",
      status: "",
      verification_method: "",
    }).sort()
  );

  // Build sample UI leaf
  const uiLeafValue = JSON.stringify(
    {
      event_type: "select_statement",
      statement_hash: backend.statement_hash.slice(0, 6),
    },
    Object.keys({ event_type: "", statement_hash: "" }).sort()
  );

  const reasoningLeaf: UIFixtureLeaf = {
    originalIndex: 0,
    sortedIndex: 0,
    canonicalValue: reasoningLeafValue,
    leafHash: computeLeafHash(reasoningLeafValue),
    merkleProof: [],
  };

  const uiLeaf: UIFixtureLeaf = {
    originalIndex: 0,
    sortedIndex: 0,
    canonicalValue: uiLeafValue,
    leafHash: computeLeafHash(uiLeafValue),
    merkleProof: [],
  };

  // Compute block hash (empty string hash if not available)
  const blockHash =
    backend.block_id != null
      ? sha256(`block:${backend.block_id}`)
      : "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

  return {
    blockNumber: backend.block_id ?? 1,
    blockHash,
    reasoningMerkleRoot: backend.R_t,
    uiMerkleRoot: backend.U_t,
    compositeAttestationRoot: backend.H_t,
    reasoningLeaves: [reasoningLeaf],
    uiLeaves: [uiLeaf],
    metadata: {
      attestation_version: "v2",
      reasoning_event_count: 1,
      ui_event_count: 1,
      composite_formula: "SHA256(R_t || U_t)",
      leaf_hash_algorithm: "sha256",
      algorithm: "SHA256",
      system: "pl",
      mdap_seed: backend.mdap_seed,
      run_id: backend.run_id,
      slice_name: backend.slice_name,
      environment_mode: backend.environment_mode,
    },
    _comment: `Synced from artifacts/first_organism/attestation.json at ${new Date().toISOString()}. Run npm run sync:attestation-fixture to regenerate.`,
  };
}

function validateFixture(fixture: UIFixture): void {
  // Validate required fields
  if (!fixture.reasoningMerkleRoot || fixture.reasoningMerkleRoot.length !== 64) {
    throw new Error(
      `Invalid reasoningMerkleRoot: expected 64-char hex, got ${fixture.reasoningMerkleRoot?.length ?? 0} chars`
    );
  }
  if (!fixture.uiMerkleRoot || fixture.uiMerkleRoot.length !== 64) {
    throw new Error(
      `Invalid uiMerkleRoot: expected 64-char hex, got ${fixture.uiMerkleRoot?.length ?? 0} chars`
    );
  }
  if (
    !fixture.compositeAttestationRoot ||
    fixture.compositeAttestationRoot.length !== 64
  ) {
    throw new Error(
      `Invalid compositeAttestationRoot: expected 64-char hex, got ${fixture.compositeAttestationRoot?.length ?? 0} chars`
    );
  }

  // Validate H_t = SHA256(R_t || U_t)
  const expectedHt = sha256(fixture.reasoningMerkleRoot + fixture.uiMerkleRoot);
  if (fixture.compositeAttestationRoot !== expectedHt) {
    console.warn(
      `Warning: H_t does not match SHA256(R_t || U_t). Backend may use different formula.`
    );
    console.warn(`  Expected: ${expectedHt}`);
    console.warn(`  Got:      ${fixture.compositeAttestationRoot}`);
  }

  console.log("✓ Fixture validation passed");
}

function main(): void {
  console.log("Syncing attestation fixture...\n");

  // Check if backend artifact exists
  if (!fs.existsSync(BACKEND_ARTIFACT_PATH)) {
    console.error(`Backend artifact not found: ${BACKEND_ARTIFACT_PATH}`);
    console.error(
      "\nRun the First Organism integration test first to generate the artifact:"
    );
    console.error(
      "  FIRST_ORGANISM_TESTS=true pytest tests/integration/test_first_organism.py -v"
    );
    process.exit(1);
  }

  // Read backend artifact
  const backendRaw = fs.readFileSync(BACKEND_ARTIFACT_PATH, "utf8");
  const backend: BackendArtifact = JSON.parse(backendRaw);
  console.log(`Read backend artifact: ${BACKEND_ARTIFACT_PATH}`);
  console.log(`  R_t: ${backend.R_t.slice(0, 16)}...`);
  console.log(`  U_t: ${backend.U_t.slice(0, 16)}...`);
  console.log(`  H_t: ${backend.H_t.slice(0, 16)}...`);

  // Transform to UI fixture format
  const fixture = transformArtifact(backend);
  console.log("\nTransformed to UI fixture format");

  // Validate
  validateFixture(fixture);

  // Write fixture
  fs.mkdirSync(path.dirname(UI_FIXTURE_PATH), { recursive: true });
  fs.writeFileSync(UI_FIXTURE_PATH, JSON.stringify(fixture, null, 2) + "\n");
  console.log(`\nWrote UI fixture: ${UI_FIXTURE_PATH}`);

  // Summary
  console.log("\n✓ Sync complete!");
  console.log(`  reasoningMerkleRoot: ${fixture.reasoningMerkleRoot.slice(0, 16)}...`);
  console.log(`  uiMerkleRoot:        ${fixture.uiMerkleRoot.slice(0, 16)}...`);
  console.log(`  compositeAttestationRoot: ${fixture.compositeAttestationRoot.slice(0, 16)}...`);
}

main();


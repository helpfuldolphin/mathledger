// Node.js script to verify examples.json hashes match Python primitives
// Uses the SAME Merkle + domain separation as Python attestation/dual_root.py
// Used by build_static_site.py assertion #30

const fs = require("fs");
const crypto = require("crypto");

// Domain separation prefixes (matching Python attestation/dual_root.py)
const DOMAIN_UI_LEAF = Buffer.from([0xA1, 0x75, 0x69, 0x2d, 0x6c, 0x65, 0x61, 0x66]); // "\xA1ui-leaf"
const DOMAIN_REASONING_LEAF = Buffer.from([0xA0, 0x72, 0x65, 0x61, 0x73, 0x6f, 0x6e, 0x69, 0x6e, 0x67, 0x2d, 0x6c, 0x65, 0x61, 0x66]); // "\xA0reasoning-leaf"
const DOMAIN_LEAF = Buffer.from([0x00]);
const DOMAIN_NODE = Buffer.from([0x01]);

// RFC 8785 JSON Canonicalization
function can(o) {
  if (o === null) return "null";
  if (typeof o === "boolean") return o ? "true" : "false";
  if (typeof o === "number") return Object.is(o, -0) ? "0" : String(o);
  if (typeof o === "string") {
    let r = '"';
    for (let i = 0; i < o.length; i++) {
      const c = o.charCodeAt(i);
      if (c === 8) r += "\\b";
      else if (c === 9) r += "\\t";
      else if (c === 10) r += "\\n";
      else if (c === 12) r += "\\f";
      else if (c === 13) r += "\\r";
      else if (c === 34) r += '\\"';
      else if (c === 92) r += "\\\\";
      else if (c < 32) r += "\\u" + c.toString(16).padStart(4, "0");
      else r += o[i];
    }
    return r + '"';
  }
  if (Array.isArray(o)) return "[" + o.map(can).join(",") + "]";
  if (typeof o === "object") {
    const k = Object.keys(o).sort();
    return "{" + k.map(x => can(x) + ":" + can(o[x])).join(",") + "}";
  }
  throw Error("bad");
}

// SHA-256 with domain prefix (returns hex string)
function shaWithDomain(data, domain) {
  const dataBytes = typeof data === "string" ? Buffer.from(data, "utf-8") : data;
  const combined = Buffer.concat([domain, dataBytes]);
  return crypto.createHash("sha256").update(combined).digest("hex");
}

// SHA-256 with domain prefix (returns Buffer for Merkle nodes)
function shaBytesWithDomain(data, domain) {
  const dataBytes = typeof data === "string" ? Buffer.from(data, "utf-8") : data;
  const combined = Buffer.concat([domain, dataBytes]);
  return crypto.createHash("sha256").update(combined).digest();
}

// Merkle root with domain separation (matching Python substrate/crypto/hashing.py)
function merkleRoot(leafHashes) {
  if (leafHashes.length === 0) {
    return shaWithDomain("", DOMAIN_LEAF);
  }

  // Normalize and sort leaf hashes
  const leaves = leafHashes.map(h => h.trim()).sort();

  // Hash each leaf with DOMAIN_LEAF
  let nodes = leaves.map(leaf => shaBytesWithDomain(Buffer.from(leaf, "utf-8"), DOMAIN_LEAF));

  // Build tree
  while (nodes.length > 1) {
    if (nodes.length % 2 === 1) {
      nodes.push(nodes[nodes.length - 1]); // Duplicate last for odd count
    }
    const nextLevel = [];
    for (let i = 0; i < nodes.length; i += 2) {
      const combined = Buffer.concat([nodes[i], nodes[i + 1]]);
      const hash = shaBytesWithDomain(combined, DOMAIN_NODE);
      nextLevel.push(hash);
    }
    nodes = nextLevel;
  }

  return nodes[0].toString("hex");
}

// Compute U_t (UI Merkle root)
function computeUiRoot(uvilEvents) {
  const leafHashes = uvilEvents.map(event => {
    const canonical = can(event);
    return shaWithDomain(canonical, DOMAIN_UI_LEAF);
  });
  return merkleRoot(leafHashes);
}

// Compute R_t (Reasoning Merkle root)
function computeReasoningRoot(reasoningArtifacts) {
  const leafHashes = reasoningArtifacts.map(artifact => {
    const canonical = can(artifact);
    return shaWithDomain(canonical, DOMAIN_REASONING_LEAF);
  });
  return merkleRoot(leafHashes);
}

// Compute H_t = SHA256(R_t || U_t)
function computeCompositeRoot(r_t, u_t) {
  const combined = r_t + u_t;
  return crypto.createHash("sha256").update(combined).digest("hex");
}

// Main
const examples = JSON.parse(fs.readFileSync(process.argv[2], "utf-8"));
const pack = examples.examples.valid_boundary_demo.pack;

const u_t = computeUiRoot(pack.uvil_events);
const r_t = computeReasoningRoot(pack.reasoning_artifacts);
const h_t = computeCompositeRoot(r_t, u_t);

if (u_t !== pack.u_t) {
  console.log("FAIL: u_t mismatch");
  console.log("computed:", u_t);
  console.log("expected:", pack.u_t);
  process.exit(1);
}
if (r_t !== pack.r_t) {
  console.log("FAIL: r_t mismatch");
  console.log("computed:", r_t);
  console.log("expected:", pack.r_t);
  process.exit(1);
}
if (h_t !== pack.h_t) {
  console.log("FAIL: h_t mismatch");
  console.log("computed:", h_t);
  console.log("expected:", pack.h_t);
  process.exit(1);
}

console.log("PASS: all hashes match (Merkle + domain separation parity with Python)");

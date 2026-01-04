#!/usr/bin/env node
/**
 * Stage-wise trace for U_t computation in JavaScript.
 *
 * This script traces every step of U_t computation to identify divergence
 * between Python and JS implementations.
 *
 * Usage:
 *   node scripts/trace_ut_computation.mjs
 */

import { createHash } from 'crypto';
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, '..');
const EXAMPLES_PATH = join(REPO_ROOT, 'site', 'v0.2.6', 'evidence-pack', 'examples.json');

// Domain separation constants (must match Python)
const DOMAIN_REASONING_LEAF = Buffer.concat([
    Buffer.from([0xA0]),
    Buffer.from('reasoning-leaf', 'utf8')
]);
const DOMAIN_UI_LEAF = Buffer.concat([
    Buffer.from([0xA1]),
    Buffer.from('ui-leaf', 'utf8')
]);
const DOMAIN_LEAF = Buffer.from([0x00]);
const DOMAIN_NODE = Buffer.from([0x01]);

// RFC 8785 JSON canonicalization
function can(o) {
    if (o === null) return 'null';
    if (typeof o === 'boolean') return o ? 'true' : 'false';
    if (typeof o === 'number') return Object.is(o, -0) ? '0' : String(o);
    if (typeof o === 'string') {
        let r = '"';
        for (let i = 0; i < o.length; i++) {
            const c = o.charCodeAt(i);
            if (c === 8) r += '\\b';
            else if (c === 9) r += '\\t';
            else if (c === 10) r += '\\n';
            else if (c === 12) r += '\\f';
            else if (c === 13) r += '\\r';
            else if (c === 34) r += '\\"';
            else if (c === 92) r += '\\\\';
            else if (c < 32) r += '\\u' + c.toString(16).padStart(4, '0');
            else r += o[i];
        }
        return r + '"';
    }
    if (Array.isArray(o)) return '[' + o.map(can).join(',') + ']';
    if (typeof o === 'object') {
        const k = Object.keys(o).sort();
        return '{' + k.map(x => can(x) + ':' + can(o[x])).join(',') + '}';
    }
    throw Error('bad');
}

// SHA256 with domain prefix (returns hex string)
function shaD(data, domain) {
    const dataBytes = typeof data === 'string' ? Buffer.from(data, 'utf8') : data;
    const combined = Buffer.concat([domain, dataBytes]);
    return createHash('sha256').update(combined).digest('hex');
}

// SHA256 with domain prefix (returns Buffer)
function shaDBytes(data, domain) {
    const dataBytes = typeof data === 'string' ? Buffer.from(data, 'utf8') : data;
    const combined = Buffer.concat([domain, dataBytes]);
    return createHash('sha256').update(combined).digest();
}

function traceUtComputation(uvilEvents) {
    const trace = {
        input_events: uvilEvents,
        event_traces: [],
        merkle_trace: null,
        final_u_t: null
    };

    const leafHashes = [];

    console.log('='.repeat(60));
    console.log('STAGE 1: Hash each UI event with DOMAIN_UI_LEAF');
    console.log('='.repeat(60));

    for (let i = 0; i < uvilEvents.length; i++) {
        const event = uvilEvents[i];
        const eventTrace = { index: i, original: event };

        // Step 1: RFC 8785 canonicalization
        const canonical = can(event);
        eventTrace.canonical_json = canonical;
        console.log(`\n[Event ${i}] RFC 8785 canonical:`);
        console.log(`  ${canonical}`);

        // Step 2: UTF-8 encode
        const canonicalBytes = Buffer.from(canonical, 'utf8');
        eventTrace.canonical_bytes_hex = canonicalBytes.toString('hex');
        console.log(`  Bytes (UTF-8): ${canonicalBytes.toString('hex')}`);

        // Step 3: Domain prefix
        eventTrace.domain_prefix_hex = DOMAIN_UI_LEAF.toString('hex');
        console.log(`  Domain prefix: ${DOMAIN_UI_LEAF.toString('hex')} (${DOMAIN_UI_LEAF})`);

        // Step 4: Full payload to hash
        const fullPayload = Buffer.concat([DOMAIN_UI_LEAF, canonicalBytes]);
        eventTrace.full_payload_hex = fullPayload.toString('hex');
        console.log(`  Full payload: ${fullPayload.toString('hex').slice(0, 40)}...`);

        // Step 5: SHA256 hash
        const leafHash = createHash('sha256').update(fullPayload).digest('hex');
        eventTrace.leaf_hash = leafHash;
        console.log(`  Leaf hash: ${leafHash}`);

        // Verify matches our shaD function
        const verifyHash = shaD(canonical, DOMAIN_UI_LEAF);
        eventTrace.shaD_result = verifyHash;
        if (leafHash !== verifyHash) {
            console.log(`  ERROR: Mismatch! ${leafHash} != ${verifyHash}`);
        } else {
            console.log(`  (verified: matches shaD)`);
        }

        leafHashes.push(leafHash);
        trace.event_traces.push(eventTrace);
    }

    console.log('\n' + '='.repeat(60));
    console.log('STAGE 2: Merkle tree construction');
    console.log('='.repeat(60));

    const merkleTrace = traceMerkleRoot(leafHashes);
    trace.merkle_trace = merkleTrace;
    trace.final_u_t = merkleTrace.final_root;

    return trace;
}

function traceMerkleRoot(leafHashes) {
    const trace = {
        input_leaf_hashes: leafHashes,
        after_sort: [],
        after_encode: [],
        levels: [],
        final_root: null
    };

    console.log(`\nInput leaf hashes (${leafHashes.length} items):`);
    for (let i = 0; i < leafHashes.length; i++) {
        console.log(`  [${i}] ${leafHashes[i]}`);
    }

    // Step 1: Sort the hex string hashes
    const sorted = [...leafHashes].sort();
    console.log('\n--- After sorting (strings) ---');
    for (let i = 0; i < sorted.length; i++) {
        trace.after_sort.push(sorted[i]);
        console.log(`  [${i}] ${sorted[i]}`);
    }

    // Step 2: UTF-8 encode each sorted hash string
    console.log('\n--- After UTF-8 encode ---');
    const encodedLeaves = [];
    for (let i = 0; i < sorted.length; i++) {
        const encoded = Buffer.from(sorted[i], 'utf8');
        encodedLeaves.push(encoded);
        trace.after_encode.push(encoded.toString('hex'));
        console.log(`  [${i}] ${encoded.toString('hex')}`);
    }

    // Step 3: Hash each leaf with DOMAIN_LEAF
    console.log('\n--- Hash with DOMAIN_LEAF ---');
    let nodes = [];
    const level0 = [];
    for (let i = 0; i < encodedLeaves.length; i++) {
        const fullPayload = Buffer.concat([DOMAIN_LEAF, encodedLeaves[i]]);
        const nodeHash = createHash('sha256').update(fullPayload).digest();
        nodes.push(nodeHash);
        level0.push({
            payload_hex: fullPayload.toString('hex'),
            hash_hex: nodeHash.toString('hex')
        });
        console.log(`  [${i}] payload: ${fullPayload.toString('hex').slice(0, 40)}...`);
        console.log(`       hash: ${nodeHash.toString('hex')}`);
    }
    trace.levels.push(level0);

    // Step 4: Build tree
    let levelNum = 1;
    while (nodes.length > 1) {
        console.log(`\n--- Level ${levelNum} (${nodes.length} -> ${Math.ceil(nodes.length / 2)}) ---`);

        if (nodes.length % 2 === 1) {
            nodes.push(nodes[nodes.length - 1]);
            console.log(`  (duplicated last node for odd count)`);
        }

        const nextLevel = [];
        const levelTrace = [];
        for (let i = 0; i < nodes.length; i += 2) {
            const combined = Buffer.concat([nodes[i], nodes[i + 1]]);
            const fullPayload = Buffer.concat([DOMAIN_NODE, combined]);
            const nodeHash = createHash('sha256').update(fullPayload).digest();
            nextLevel.push(nodeHash);
            levelTrace.push({
                left_hash: nodes[i].toString('hex'),
                right_hash: nodes[i + 1].toString('hex'),
                result_hash: nodeHash.toString('hex')
            });
            console.log(`  [${Math.floor(i / 2)}] L: ${nodes[i].toString('hex').slice(0, 16)}... R: ${nodes[i + 1].toString('hex').slice(0, 16)}...`);
            console.log(`       Result: ${nodeHash.toString('hex')}`);
        }

        trace.levels.push(levelTrace);
        nodes = nextLevel;
        levelNum++;
    }

    const finalRoot = nodes[0].toString('hex');
    trace.final_root = finalRoot;
    console.log(`\nFinal U_t: ${finalRoot}`);

    return trace;
}

function main() {
    // Load test data
    const examples = JSON.parse(readFileSync(EXAMPLES_PATH, 'utf8'));
    const validPack = examples.examples.valid_boundary_demo.pack;

    const uvilEvents = validPack.uvil_events;
    const expectedUt = validPack.u_t;

    console.log('='.repeat(60));
    console.log('U_t COMPUTATION TRACE (JavaScript)');
    console.log('='.repeat(60));
    console.log(`Expected U_t: ${expectedUt}`);
    console.log();

    const trace = traceUtComputation(uvilEvents);

    console.log('\n' + '='.repeat(60));
    console.log('VERIFICATION');
    console.log('='.repeat(60));
    console.log(`Expected: ${expectedUt}`);
    console.log(`Computed: ${trace.final_u_t}`);
    if (trace.final_u_t === expectedUt) {
        console.log('STATUS: MATCH!');
    } else {
        console.log('STATUS: MISMATCH!');
    }

    // Save trace for comparison
    const tmpDir = join(REPO_ROOT, 'tmp');
    try { mkdirSync(tmpDir, { recursive: true }); } catch (e) {}
    const tracePath = join(tmpDir, 'ut_trace_js.json');
    writeFileSync(tracePath, JSON.stringify(trace, null, 2));
    console.log(`\nTrace saved to: ${tracePath}`);
}

main();

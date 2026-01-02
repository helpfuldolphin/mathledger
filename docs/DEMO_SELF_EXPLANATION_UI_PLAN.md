# Demo Self-Explanation UI Integration Plan

**Version**: 0.1
**Date**: 2026-01-02
**Purpose**: Specify UI integration points for governance explanations

---

## Overview

This document specifies exactly where and how governance explanations appear in the demo UI. All copy follows three rules:

1. **Never claim correctness beyond enforcement** - Do not say "verified correct"; say "returned VERIFIED by validator"
2. **Never anthropomorphize** - Do not say "the system believes"; say "the system computed"
3. **Never suggest safety/alignment** - Do not say "safe" or "aligned"; say "auditable" or "replayable"

---

## UI Integration Points

### 1. Framing Box (Top of Page)

**Location**: `.framing` div, visible on page load

**Current Copy**:
```html
<p><strong>The system does not decide what is true. It decides what is justified under a declared verification route.</strong></p>
<p>This demo will stop more often than you expect. It reports what it cannot verify.</p>
<p>If you are looking for a system that always has an answer, this demo is not it.</p>
```

**Enhancement**: Add expandable detail section:

```html
<details style="margin-top:0.75rem; font-size:0.8rem;">
    <summary style="cursor:pointer;">What does "justified" mean?</summary>
    <p style="margin-top:0.5rem; padding-left:1rem; color:#666;">
        A claim is justified when it passes through attestation with a declared trust class.
        VERIFIED means a validator confirmed the claim. ABSTAINED means no validator could
        confirm or refute it. Neither outcome implies truth — only what the system could verify.
    </p>
</details>
```

**File**: `demo/app.py` (HTML_CONTENT, line ~428-432)

---

### 2. Draft Phase (Exploration Panel)

**Location**: `.exploration-panel`, after scenario selection

**State**: Draft claims visible, proposal_id shown

**Current Copy**:
```html
<p class="warning">This ID is exploration-only. It will NOT appear in any committed data.</p>
```

**Enhancement**: Add tooltip explanation for each trust class selector:

```html
<select class="trust-select" title="FV: Claim requires formal proof (returns ABSTAINED in v0)&#10;MV: Claim is mechanically checkable (arithmetic only in v0)&#10;PA: User attests correctness (returns ABSTAINED)&#10;ADV: Exploration only, excluded from R_t">
```

**Additional copy below claims**:
```html
<p class="note" style="margin-top:0.75rem;">
    <strong>Trust class determines verification route, not correctness.</strong>
    Selecting MV does not make a claim verified — it declares that mechanical verification
    should be attempted. If no validator exists, the outcome is ABSTAINED.
</p>
```

**File**: `demo/app.py` (renderDraftClaims function, line ~700-727)

---

### 3. Commit Transition

**Location**: Click "Commit to Authority" button

**Current Behavior**: Shows committed_id and hashes

**Enhancement**: Add transition explanation:

```html
<div class="transition-note" style="background:#fff3e0; padding:0.75rem; margin:0.5rem 0; font-size:0.85rem;">
    <strong>Transition: Exploration → Authority</strong><br>
    The random proposal_id (<code>proposal_xxx</code>) is discarded.
    The committed_id (<code>sha256:xxx</code>) is derived from claim content.
    This ID is immutable — changing the claims would produce a different ID.
</div>
```

**File**: `demo/app.py` (commitUVIL function response handling, line ~751-758)

---

### 4. Verification Outcome Display

**Location**: `.outcome-section` in authority panel

#### 4a. ABSTAINED Outcome

**Current Copy**:
```html
<p class="note" style="margin-top:0.75rem; font-size:0.8rem; color:#666;">
    <strong>ABSTAINED is not failure.</strong> It means: no verifier exists for this claim in v0,
    so the system refuses to assert correctness. The authority stream shows commitments and boundaries, not truth.
</p>
```

**Enhancement**: Make this more prominent for ABSTAINED:

```html
<div class="outcome-explanation-detail" style="margin-top:1rem; padding:1rem; background:#fffde7; border:1px solid #ffc107; font-size:0.85rem;">
    <strong>Why ABSTAINED?</strong>
    <ul style="margin:0.5rem 0 0 1.5rem; padding:0;">
        <li>v0 has no formal verifier (FV claims always abstain)</li>
        <li>MV validator handles arithmetic only (<code>a op b = c</code>)</li>
        <li>PA claims require human attestation, not mechanical verification</li>
    </ul>
    <p style="margin-top:0.75rem; margin-bottom:0;">
        <strong>ABSTAINED is recorded in R_t.</strong> It is a first-class outcome,
        not a missing value. Downstream systems cannot ignore it.
    </p>
</div>
```

**File**: `demo/app.py` (runVerification function, line ~817-827)

#### 4b. VERIFIED Outcome

**Copy**:
```html
<div class="outcome-explanation-detail" style="margin-top:1rem; padding:1rem; background:#e8f5e9; border:1px solid #4caf50; font-size:0.85rem;">
    <strong>VERIFIED by arithmetic validator</strong>
    <p style="margin:0.5rem 0 0 0;">
        The MV validator parsed this claim as <code>a op b = c</code> and computed
        that the equation holds. This is the only validator in v0.
    </p>
    <p style="margin-top:0.5rem; margin-bottom:0; color:#666;">
        VERIFIED means the validator returned true. It does not mean the claim is universally correct.
    </p>
</div>
```

#### 4c. REFUTED Outcome

**Copy**:
```html
<div class="outcome-explanation-detail" style="margin-top:1rem; padding:1rem; background:#ffebee; border:1px solid #f44336; font-size:0.85rem;">
    <strong>REFUTED by arithmetic validator</strong>
    <p style="margin:0.5rem 0 0 0;">
        The MV validator parsed this claim as <code>a op b = c</code> and computed
        that the equation does not hold.
    </p>
    <p style="margin-top:0.5rem; margin-bottom:0; color:#666;">
        REFUTED means the validator returned false. The claim is recorded with this outcome.
    </p>
</div>
```

---

### 5. Evidence Pack Download

**Location**: `.evidence-pack-section`

**Current Copy**:
```html
<p class="note" style="margin-bottom:0.75rem;">
    Download the evidence pack to independently verify attestation hashes.
    Replay verification recomputes U_t, R_t, H_t locally with no external calls.
</p>
```

**Enhancement**: Add explanation of what the pack contains:

```html
<details style="margin-bottom:0.75rem; font-size:0.8rem;">
    <summary style="cursor:pointer;">What is an evidence pack?</summary>
    <div style="margin-top:0.5rem; padding-left:1rem; color:#666;">
        <p style="margin:0.25rem 0;">The evidence pack is a self-contained JSON file containing:</p>
        <ul style="margin:0.5rem 0 0.5rem 1rem; padding:0;">
            <li><code>uvil_events</code> — User interaction events (hashed into U_t)</li>
            <li><code>reasoning_artifacts</code> — Claims with outcomes (hashed into R_t)</li>
            <li><code>u_t</code>, <code>r_t</code>, <code>h_t</code> — Attestation roots</li>
            <li><code>replay_instructions</code> — Commands to reproduce hashes</li>
        </ul>
        <p style="margin:0;">
            Anyone with the evidence pack can recompute the hashes independently.
            If replay produces different hashes, the pack is invalid.
        </p>
    </div>
</details>
```

**File**: `demo/app.py` (HTML_CONTENT, line ~570-580)

---

### 6. Hash Display Sections

**Location**: `.hash-section` elements

**Current Copy**:
```html
<p class="note" style="margin-top:0.5rem; font-size:0.7rem;">
    R_t commits to authority-bearing artifacts only (no ADV). In v0, these are demo placeholders marked <code>v0_mock:true</code> — not proofs.
</p>
```

**Enhancement**: Add hover explanations:

```html
<div class="hash-row">
    <span class="hash-label" title="U_t is the Merkle root of all user interaction events. It commits to what the user saw and did.">U_t (UI):</span>
    <span id="hash-ut">-</span>
</div>
<div class="hash-row">
    <span class="hash-label" title="R_t is the Merkle root of reasoning artifacts. ADV claims are excluded. Only MV, PA, FV claims enter this root.">R_t (Reasoning):</span>
    <span id="hash-rt">-</span>
</div>
<div class="hash-row">
    <span class="hash-label" title="H_t = SHA256(R_t || U_t). This is the composite epoch root that binds UI and reasoning together.">H_t (Composite):</span>
    <span id="hash-ht">-</span>
</div>
```

**File**: `demo/app.py` (HTML_CONTENT, line ~536-543)

---

### 7. Boundary Demo Section

**Location**: `.boundary-demo` section

**Current Conclusion**:
```html
<p><strong>Same claim text, different trust class → different outcome. Same trust class, different truth → VERIFIED vs REFUTED.</strong></p>
```

**Enhancement**: Add expandable breakdown:

```html
<details style="margin-top:1rem; font-size:0.8rem; color:#aaa;">
    <summary style="cursor:pointer;">What does this prove?</summary>
    <div style="margin-top:0.5rem; padding-left:1rem;">
        <p style="margin:0.25rem 0;">
            <strong>Outcome is determined by:</strong> trust class + validator + claim truth
        </p>
        <ul style="margin:0.5rem 0 0 1rem; padding:0;">
            <li>ADV: Always excluded from R_t (no verification attempted)</li>
            <li>PA: Enters R_t but no validator exists → ABSTAINED</li>
            <li>MV + true arithmetic: Validator confirms → VERIFIED</li>
            <li>MV + false arithmetic: Validator refutes → REFUTED</li>
        </ul>
        <p style="margin-top:0.5rem; margin-bottom:0;">
            The system does not infer correctness from content. It applies declared verification routes.
        </p>
    </div>
</details>
```

**File**: `demo/app.py` (HTML_CONTENT, line ~469-472)

---

### 8. ADV Exclusion Badge

**Location**: Claim items with ADV trust class

**Current Copy**:
```html
<span class="excluded-badge">EXCLUDED FROM R_t</span>
```

**Enhancement**: Add tooltip:

```html
<span class="excluded-badge" title="ADV (Advisory) claims are exploration-only. They do not enter R_t and cannot influence authority attestation. Use ADV for speculation, notes, or hypotheses.">EXCLUDED FROM R_t</span>
```

**File**: `demo/app.py` (renderDraftClaims and renderAuthorityClaims functions)

---

### 9. Error States

**Location**: `.error` display element

**Governance Error Copy Templates**:

```javascript
// Trust-Class Monotonicity Violation
const errorCopy = {
    "TRUST_CLASS_MONOTONICITY_VIOLATION":
        "Cannot change trust class of committed claim. Original: {from}, Attempted: {to}. " +
        "To use a different trust class, create a new claim (which produces a new claim_id).",

    "SILENT_AUTHORITY_VIOLATION":
        "Cannot produce evidence pack: attestation verification failed. " +
        "Claimed H_t does not match computed H_t. This indicates tampering or corruption.",

    "DOUBLE_COMMIT":
        "This partition has already been committed. Duplicate commit rejected."
};
```

**File**: `demo/app.py` (showError function, enhance with structured error handling)

---

## Implementation Checklist

| Section | File | Lines | Status |
|---------|------|-------|--------|
| Framing box expandable | `demo/app.py` | ~428-432 | TODO |
| Trust class tooltips | `demo/app.py` | ~700-727 | TODO |
| Transition note | `demo/app.py` | ~751-758 | TODO |
| ABSTAINED explanation | `demo/app.py` | ~817-827 | TODO |
| Evidence pack details | `demo/app.py` | ~570-580 | TODO |
| Hash label tooltips | `demo/app.py` | ~536-543 | TODO |
| Boundary demo breakdown | `demo/app.py` | ~469-472 | TODO |
| ADV badge tooltip | `demo/app.py` | ~720, ~785 | TODO |
| Error templates | `demo/app.py` | ~873-877 | TODO |

---

## Copy Style Guide

### Do Write:
- "The system computed X"
- "The validator returned VERIFIED"
- "This outcome is recorded in R_t"
- "Replay verification recomputes hashes"
- "ABSTAINED means no verifier could confirm or refute"

### Do Not Write:
- "The system believes X"
- "This is correct/true"
- "The system safely handled..."
- "AI-powered verification"
- "Smart validation"

### Outcome Descriptions:

| Outcome | Description |
|---------|-------------|
| VERIFIED | "The validator confirmed the claim" |
| REFUTED | "The validator disproved the claim" |
| ABSTAINED | "No verifier could confirm or refute" |

### Trust Class Descriptions:

| Class | Description |
|-------|-------------|
| FV | "Requires formal proof (not implemented in v0)" |
| MV | "Mechanically checkable (arithmetic validator only in v0)" |
| PA | "User-attested (no mechanical verification)" |
| ADV | "Exploration only, excluded from authority stream" |

---

## References

- `docs/HOW_THE_DEMO_EXPLAINS_ITSELF.md` - Existing explanation prose
- `docs/HOW_TO_APPROACH_THIS_DEMO.md` - Framing document
- `docs/V0_SYSTEM_BOUNDARY_MEMO.md` - System boundary definitions

---

**Author**: Claude A (v0.1 UI Integration Plan)
**Date**: 2026-01-02

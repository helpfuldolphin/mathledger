"""
MathLedger UVIL v0 Demo

Front-facing demo with scenario selector and split panels.
Serves the UVIL API and interactive HTML interface.

Run: uv run python demo/app.py
Open: http://localhost:8000
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

from backend.api.uvil import router as uvil_router

app = FastAPI(
    title="MathLedger Demo",
    description="UVIL v0 + Trust Classes v0: Epistemic Governance Demo",
    version="0.1.0",
)

# Mount UVIL API
app.include_router(uvil_router)

# Predefined scenarios (mirrors fixtures from harness)
SCENARIOS = {
    "mv_only": {
        "name": "MV Only",
        "description": "Single mechanically-validated claim",
        "task_text": "Prove that addition is commutative for natural numbers",
        "claims": [
            {"claim_text": "forall a b : Nat, a + b = b + a", "trust_class": "MV", "rationale": "Commutativity - mechanically checkable"}
        ]
    },
    "mixed_mv_adv": {
        "name": "Mixed MV + ADV",
        "description": "One authority-bearing, one exploration-only",
        "task_text": "Prove commutativity and consider generalizations",
        "claims": [
            {"claim_text": "forall a b : Nat, a + b = b + a", "trust_class": "MV", "rationale": "Commutativity - mechanically checkable"},
            {"claim_text": "This likely generalizes to arbitrary rings", "trust_class": "ADV", "rationale": "Speculation about generalization"}
        ]
    },
    "pa_only": {
        "name": "PA Only (User Attestation)",
        "description": "User-attested claim, not mechanically verified",
        "task_text": "Attest that requirement REQ-001 is satisfied",
        "claims": [
            {"claim_text": "Requirement REQ-001 is satisfied by implementation in module X", "trust_class": "PA", "rationale": "User attestation based on manual review"}
        ]
    },
    "adv_only": {
        "name": "ADV Only (Exploration)",
        "description": "All claims are advisory - nothing enters authority stream",
        "task_text": "Speculate about Navier-Stokes",
        "claims": [
            {"claim_text": "The Navier-Stokes equations probably have smooth solutions", "trust_class": "ADV", "rationale": "Pure speculation"},
            {"claim_text": "Turbulence might be related to strange attractors", "trust_class": "ADV", "rationale": "Another guess"}
        ]
    },
    "underdetermined": {
        "name": "Underdetermined (Open Problem)",
        "description": "System correctly stops - cannot verify open problems",
        "task_text": "Prove existence and smoothness of Navier-Stokes solutions in 3D",
        "claims": [
            {"claim_text": "Existence of weak solutions follows from energy estimates", "trust_class": "ADV", "rationale": "Standard but unverified"},
            {"claim_text": "Smoothness in 3D remains open (Millennium Prize)", "trust_class": "ADV", "rationale": "Open problem"},
            {"claim_text": "Partial regularity results exist (Caffarelli-Kohn-Nirenberg)", "trust_class": "ADV", "rationale": "Reference to literature"}
        ]
    }
}

# HTML Frontend with split panels
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MathLedger Demo</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            background: #f5f5f5;
            color: #1a1a1a;
            line-height: 1.5;
            font-size: 14px;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 1.5rem; }
        h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 1rem; }

        /* Framing box */
        .framing {
            background: #fff;
            border: 1px solid #ddd;
            padding: 1rem;
            margin-bottom: 1.5rem;
            font-size: 0.85rem;
        }
        .framing p { margin: 0.5rem 0; }
        .framing p:last-child { font-style: italic; margin-top: 0.75rem; }

        /* Scenario selector */
        .scenario-bar {
            background: #fff;
            border: 1px solid #ddd;
            padding: 1rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
        }
        .scenario-bar label { font-weight: 600; }
        .scenario-bar select {
            padding: 0.5rem;
            font-size: 0.9rem;
            min-width: 200px;
        }
        .scenario-bar button {
            background: #2a2a2a;
            color: #fff;
            border: none;
            padding: 0.5rem 1rem;
            cursor: pointer;
        }
        .scenario-bar button:hover { background: #444; }
        .scenario-desc { font-size: 0.85rem; color: #666; }

        /* Split panel layout */
        .panels {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        @media (max-width: 900px) {
            .panels { grid-template-columns: 1fr; }
        }

        .panel {
            background: #fff;
            border: 1px solid #ddd;
            padding: 1rem;
        }
        .panel-header {
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding-bottom: 0.75rem;
            margin-bottom: 0.75rem;
            border-bottom: 1px solid #eee;
        }

        /* Exploration panel */
        .exploration-panel { border-left: 4px solid #888; }
        .exploration-panel .panel-header { color: #666; }

        /* Authority panel */
        .authority-panel { border-left: 4px solid #2a2a2a; }
        .authority-panel .panel-header { color: #1a1a1a; }

        /* Claims */
        .claim-item {
            background: #fafafa;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border: 1px solid #eee;
            font-size: 0.85rem;
        }
        .claim-text {
            font-family: monospace;
            margin-bottom: 0.5rem;
            word-break: break-word;
        }
        .claim-text input {
            width: 100%;
            padding: 0.4rem;
            font-family: monospace;
            font-size: 0.85rem;
            border: 1px solid #ccc;
        }
        .claim-meta { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
        .trust-badge {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 2px;
        }
        .trust-fv { background: #e3f2fd; color: #1565c0; }
        .trust-mv { background: #e8f5e9; color: #2e7d32; }
        .trust-pa { background: #fff3e0; color: #ef6c00; }
        .trust-adv { background: #f5f5f5; color: #757575; }

        .trust-select {
            padding: 0.3rem;
            font-size: 0.8rem;
            border: 1px solid #ccc;
        }

        .excluded-badge {
            background: #ffebee;
            color: #c62828;
            padding: 0.2rem 0.5rem;
            font-size: 0.7rem;
            font-weight: 600;
        }
        .included-badge {
            background: #e8f5e9;
            color: #2e7d32;
            padding: 0.2rem 0.5rem;
            font-size: 0.7rem;
            font-weight: 600;
        }

        /* Hash display */
        .hash-section {
            background: #f8f8f8;
            padding: 0.75rem;
            margin-top: 1rem;
            font-family: monospace;
            font-size: 0.7rem;
            word-break: break-all;
        }
        .hash-row { margin: 0.25rem 0; }
        .hash-label { font-weight: 600; color: #444; }

        /* Outcome */
        .outcome-section {
            margin-top: 1rem;
            padding: 1rem;
            background: #fff3e0;
        }
        .outcome-header {
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }
        .outcome-explanation {
            font-size: 0.85rem;
            color: #666;
        }

        /* Actions */
        .actions {
            margin-top: 1rem;
            display: flex;
            gap: 0.5rem;
        }
        .actions button {
            background: #2a2a2a;
            color: #fff;
            border: none;
            padding: 0.6rem 1.2rem;
            cursor: pointer;
            font-size: 0.85rem;
        }
        .actions button:hover { background: #444; }
        .actions button:disabled { background: #999; cursor: not-allowed; }
        .actions button.secondary {
            background: #fff;
            color: #2a2a2a;
            border: 1px solid #2a2a2a;
        }

        /* Status */
        .status { font-size: 0.8rem; color: #666; margin-top: 0.5rem; }
        .error { color: #c00; background: #fee; padding: 0.5rem; margin: 0.5rem 0; font-size: 0.85rem; }
        .hidden { display: none; }

        /* Notes */
        .note { font-size: 0.8rem; color: #888; margin-top: 0.5rem; }
        .warning { font-size: 0.8rem; color: #c00; margin-top: 0.5rem; font-weight: 600; }

        /* Footer */
        .footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #ddd;
            font-size: 0.8rem;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>MathLedger Demo</h1>

        <div class="framing">
            <p>This demo will stop more often than you expect. It reports what it cannot verify.</p>
            <p>The goal is not to be impressive. The goal is to be legible.</p>
            <p>If you are looking for a system that always has an answer, this demo is not it.</p>
        </div>

        <!-- Scenario selector -->
        <div class="scenario-bar">
            <label>Scenario:</label>
            <select id="scenario-select" onchange="loadScenario()">
                <option value="">-- Select a scenario --</option>
                <option value="mv_only">MV Only (Mechanically Validated)</option>
                <option value="mixed_mv_adv">Mixed MV + ADV</option>
                <option value="pa_only">PA Only (User Attestation)</option>
                <option value="adv_only">ADV Only (Exploration)</option>
                <option value="underdetermined">Underdetermined (Open Problem)</option>
                <option value="custom">Custom Input</option>
            </select>
            <span id="scenario-desc" class="scenario-desc"></span>
        </div>

        <!-- Split panels -->
        <div class="panels">
            <!-- Left: Exploration Stream -->
            <div class="panel exploration-panel">
                <div class="panel-header">Exploration Stream (Not Authority)</div>

                <div id="exploration-content">
                    <p class="note">Select a scenario or enter custom input.</p>
                </div>

                <div id="custom-input" class="hidden">
                    <label>Problem statement:</label>
                    <textarea id="problem-input" style="width:100%; min-height:60px; margin:0.5rem 0; padding:0.5rem; font-size:0.85rem;" placeholder="Enter a problem or claim..."></textarea>
                    <button onclick="proposePartition()">Generate Draft</button>
                </div>

                <div id="draft-section" class="hidden">
                    <p class="note" style="margin-bottom:0.5rem;">
                        <strong>Proposal ID:</strong> <code id="proposal-id"></code>
                    </p>
                    <p class="warning">This ID is exploration-only. It will NOT appear in any committed data.</p>

                    <div id="draft-claims"></div>

                    <div class="actions">
                        <button id="btn-commit" onclick="commitUVIL()">Commit to Authority</button>
                    </div>
                </div>
            </div>

            <!-- Right: Authority Stream -->
            <div class="panel authority-panel">
                <div class="panel-header">Authority Stream (Bound)</div>

                <div id="authority-empty">
                    <p class="note">Nothing committed yet. Authority stream is empty.</p>
                </div>

                <div id="authority-content" class="hidden">
                    <p class="note" style="margin-bottom:0.5rem;">
                        <strong>Committed ID:</strong> <code id="committed-id"></code>
                    </p>
                    <p class="note">This ID is derived from content. It is immutable.</p>

                    <div id="authority-claims"></div>

                    <div class="hash-section">
                        <div class="hash-row"><span class="hash-label">U_t (UI):</span> <span id="hash-ut">-</span></div>
                        <div class="hash-row"><span class="hash-label">R_t (Reasoning):</span> <span id="hash-rt">-</span></div>
                        <div class="hash-row"><span class="hash-label">H_t (Composite):</span> <span id="hash-ht">-</span></div>
                        <p class="note" style="margin-top:0.5rem; font-size:0.7rem;">
                            R_t commits to authority-bearing artifacts only (no ADV). In v0, these are demo placeholders marked <code>v0_mock:true</code> — not proofs.
                        </p>
                    </div>

                    <div class="actions">
                        <button id="btn-verify" onclick="runVerification()">Run Verification</button>
                    </div>
                </div>

                <div id="result-section" class="hidden">
                    <div class="outcome-section">
                        <div class="outcome-header" id="outcome-text">ABSTAINED</div>
                        <div class="outcome-explanation" id="outcome-explanation"></div>
                        <p class="note" style="margin-top:0.75rem; font-size:0.8rem; color:#666;">
                            <strong>ABSTAINED is not failure.</strong> It means: no verifier exists for this claim in v0,
                            so the system refuses to assert correctness. The authority stream shows commitments and boundaries, not truth.
                        </p>
                    </div>

                    <div id="authority-breakdown" style="margin-top:1rem;"></div>

                    <div class="hash-section">
                        <div class="hash-row"><span class="hash-label">Final U_t:</span> <span id="final-ut">-</span></div>
                        <div class="hash-row"><span class="hash-label">Final R_t:</span> <span id="final-rt">-</span></div>
                        <div class="hash-row"><span class="hash-label">Final H_t:</span> <span id="final-ht">-</span></div>
                    </div>

                    <div class="actions">
                        <button class="secondary" onclick="reset()">Start Over</button>
                    </div>
                </div>
            </div>
        </div>

        <div id="error-display" class="error hidden"></div>
        <div id="status-display" class="status"></div>

        <div class="footer">
            v0 Demo | Governance substrate only | No verification implemented | <a href="/docs/V0_LOCK.md">Scope Lock</a>
        </div>
    </div>

    <script>
        const SCENARIOS = """ + str(SCENARIOS).replace("'", '"') + """;

        let currentProposalId = null;
        let currentCommittedId = null;
        let editedClaims = [];
        let currentScenario = null;

        function loadScenario() {
            const select = document.getElementById('scenario-select');
            const scenarioKey = select.value;
            const descEl = document.getElementById('scenario-desc');

            reset();

            if (!scenarioKey) {
                descEl.textContent = '';
                return;
            }

            if (scenarioKey === 'custom') {
                descEl.textContent = 'Enter your own problem and claims';
                document.getElementById('custom-input').classList.remove('hidden');
                document.getElementById('exploration-content').classList.add('hidden');
                return;
            }

            const scenario = SCENARIOS[scenarioKey];
            if (!scenario) return;

            currentScenario = scenario;
            descEl.textContent = scenario.description;

            // Auto-propose with scenario
            proposeWithScenario(scenario);
        }

        async function proposeWithScenario(scenario) {
            setStatus('Generating proposal...');

            try {
                const response = await fetch('/uvil/propose_partition', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ problem_statement: scenario.task_text })
                });

                if (!response.ok) throw new Error('Request failed');

                const data = await response.json();
                currentProposalId = data.proposal_id;

                // Use scenario claims instead of generated ones
                editedClaims = scenario.claims.map(c => ({...c}));

                renderDraftClaims();
                document.getElementById('proposal-id').textContent = currentProposalId;
                document.getElementById('exploration-content').classList.add('hidden');
                document.getElementById('draft-section').classList.remove('hidden');
                setStatus('');

            } catch (e) {
                showError(e.message);
            }
        }

        async function proposePartition() {
            const input = document.getElementById('problem-input').value.trim();
            if (!input) {
                showError('Please enter a problem statement.');
                return;
            }

            setStatus('Generating proposal...');

            try {
                const response = await fetch('/uvil/propose_partition', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ problem_statement: input })
                });

                if (!response.ok) throw new Error('Request failed');

                const data = await response.json();
                currentProposalId = data.proposal_id;
                editedClaims = data.claims.map(c => ({
                    claim_text: c.claim_text,
                    trust_class: c.suggested_trust_class,
                    rationale: c.rationale || ''
                }));

                renderDraftClaims();
                document.getElementById('proposal-id').textContent = currentProposalId;
                document.getElementById('custom-input').classList.add('hidden');
                document.getElementById('draft-section').classList.remove('hidden');
                setStatus('');

            } catch (e) {
                showError(e.message);
            }
        }

        function renderDraftClaims() {
            const container = document.getElementById('draft-claims');
            container.innerHTML = '';

            editedClaims.forEach((claim, idx) => {
                const div = document.createElement('div');
                div.className = 'claim-item';
                div.innerHTML = `
                    <div class="claim-text">
                        <input type="text" value="${escapeHtml(claim.claim_text)}"
                               onchange="editedClaims[${idx}].claim_text = this.value">
                    </div>
                    <div class="claim-meta">
                        <select class="trust-select" onchange="editedClaims[${idx}].trust_class = this.value; renderDraftClaims();">
                            <option value="ADV" ${claim.trust_class === 'ADV' ? 'selected' : ''}>ADV</option>
                            <option value="PA" ${claim.trust_class === 'PA' ? 'selected' : ''}>PA</option>
                            <option value="MV" ${claim.trust_class === 'MV' ? 'selected' : ''}>MV</option>
                            <option value="FV" ${claim.trust_class === 'FV' ? 'selected' : ''}>FV</option>
                        </select>
                        <span class="trust-badge trust-${claim.trust_class.toLowerCase()}">${claim.trust_class}</span>
                        ${claim.trust_class === 'ADV'
                            ? '<span class="excluded-badge">EXCLUDED FROM R_t</span>'
                            : '<span class="included-badge">ENTERS R_t</span>'}
                    </div>
                `;
                container.appendChild(div);
            });
        }

        async function commitUVIL() {
            if (!currentProposalId || editedClaims.length === 0) return;

            document.getElementById('btn-commit').disabled = true;
            setStatus('Committing...');

            try {
                const response = await fetch('/uvil/commit_uvil', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        proposal_id: currentProposalId,
                        edited_claims: editedClaims,
                        user_fingerprint: 'demo_user'
                    })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Commit failed');
                }

                const data = await response.json();
                currentCommittedId = data.committed_partition_id;

                // Update authority panel
                document.getElementById('committed-id').textContent = currentCommittedId;
                document.getElementById('hash-ut').textContent = data.u_t;
                document.getElementById('hash-rt').textContent = data.r_t;
                document.getElementById('hash-ht').textContent = data.h_t;

                renderAuthorityClaims();

                document.getElementById('authority-empty').classList.add('hidden');
                document.getElementById('authority-content').classList.remove('hidden');
                setStatus('Committed. Ready to verify.');

            } catch (e) {
                showError(e.message);
                document.getElementById('btn-commit').disabled = false;
            }
        }

        function renderAuthorityClaims() {
            const container = document.getElementById('authority-claims');
            container.innerHTML = '';

            editedClaims.forEach(claim => {
                const isAuthority = claim.trust_class !== 'ADV';
                const div = document.createElement('div');
                div.className = 'claim-item';
                div.style.opacity = isAuthority ? '1' : '0.5';
                div.innerHTML = `
                    <div class="claim-text">${escapeHtml(claim.claim_text)}</div>
                    <div class="claim-meta">
                        <span class="trust-badge trust-${claim.trust_class.toLowerCase()}">${claim.trust_class}</span>
                        ${isAuthority
                            ? '<span class="included-badge">IN AUTHORITY STREAM</span>'
                            : '<span class="excluded-badge">EXCLUDED (ADV)</span>'}
                    </div>
                `;
                container.appendChild(div);
            });
        }

        async function runVerification() {
            if (!currentCommittedId) return;

            document.getElementById('btn-verify').disabled = true;
            setStatus('Running verification...');

            try {
                const response = await fetch('/uvil/run_verification', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ committed_partition_id: currentCommittedId })
                });

                if (!response.ok) throw new Error('Verification failed');

                const data = await response.json();

                // Show result
                document.getElementById('outcome-text').textContent = data.outcome;
                document.getElementById('outcome-explanation').textContent =
                    data.authority_basis.explanation;

                // Breakdown
                const breakdown = document.getElementById('authority-breakdown');
                breakdown.innerHTML = `
                    <p><strong>Authority Basis:</strong></p>
                    <ul style="margin:0.5rem 0; padding-left:1.5rem; font-size:0.85rem;">
                        <li>FV claims: ${data.authority_basis.fv_count}</li>
                        <li>MV claims: ${data.authority_basis.mv_count}</li>
                        <li>PA claims: ${data.authority_basis.pa_count}</li>
                        <li>ADV claims (excluded): ${data.authority_basis.adv_count}</li>
                    </ul>
                    <p class="note">Mechanically verified: ${data.authority_basis.mechanically_verified ? 'Yes' : 'No (v0 has no verifier)'}</p>
                `;

                // Final hashes
                document.getElementById('final-ut').textContent = data.attestation.u_t;
                document.getElementById('final-rt').textContent = data.attestation.r_t;
                document.getElementById('final-ht').textContent = data.attestation.h_t;

                document.getElementById('authority-content').classList.add('hidden');
                document.getElementById('result-section').classList.remove('hidden');
                setStatus('');

            } catch (e) {
                showError(e.message);
                document.getElementById('btn-verify').disabled = false;
            }
        }

        function reset() {
            currentProposalId = null;
            currentCommittedId = null;
            editedClaims = [];
            currentScenario = null;

            document.getElementById('exploration-content').classList.remove('hidden');
            document.getElementById('custom-input').classList.add('hidden');
            document.getElementById('draft-section').classList.add('hidden');
            document.getElementById('draft-claims').innerHTML = '';

            document.getElementById('authority-empty').classList.remove('hidden');
            document.getElementById('authority-content').classList.add('hidden');
            document.getElementById('result-section').classList.add('hidden');

            document.getElementById('btn-commit').disabled = false;
            document.getElementById('btn-verify').disabled = false;

            document.getElementById('problem-input').value = '';
            hideError();
            setStatus('');
        }

        function setStatus(msg) {
            document.getElementById('status-display').textContent = msg;
        }

        function showError(msg) {
            const el = document.getElementById('error-display');
            el.textContent = msg;
            el.classList.remove('hidden');
        }

        function hideError() {
            document.getElementById('error-display').classList.add('hidden');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the demo frontend."""
    return HTML_CONTENT


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/scenarios")
async def get_scenarios():
    """Return available scenarios for the demo."""
    return SCENARIOS


if __name__ == "__main__":
    print("=" * 60)
    print("MathLedger UVIL v0 Demo")
    print("=" * 60)
    print()
    print("Open in browser: http://localhost:8000")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)

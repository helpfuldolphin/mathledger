#!/usr/bin/env python3
"""
Generate v0.2.6 Verifier with Cryptographic Parity

This script creates the v0.2.6 evidence pack verifier with proper
domain-separated Merkle tree construction that matches Python.

Usage:
    uv run python scripts/generate_v026_verifier.py
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = REPO_ROOT / "site" / "v0.2.6" / "evidence-pack" / "verify" / "index.html"

# HTML template with proper cryptographic implementation
HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evidence Pack Verifier - MathLedger v0.2.6</title>
<style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:-apple-system,BlinkMacSystemFont,monospace;background:#f5f5f5;line-height:1.6}.container{max-width:900px;margin:0 auto;padding:2rem}.banner{background:#fff;border:1px solid #ddd;border-left:4px solid #2e7d32;padding:1rem;margin-bottom:1.5rem}.status{font-weight:600;color:#2e7d32}code{background:#f0f0f0;padding:0.15em 0.35em}h1{font-size:1.4rem;margin-bottom:1rem}h2{font-size:1.2rem;margin:1.5rem 0 0.75rem;border-bottom:1px solid #ddd}p{margin:0.75rem 0}.info{background:#fff;border:1px solid #ddd;padding:1rem;margin:1rem 0;border-left:4px solid #f57c00}.vbox{background:#fff;border:1px solid #ddd;padding:1.5rem;margin:1rem 0}.result{padding:1rem;margin:1rem 0;font-family:monospace}.pass{background:#e8f5e9;border-left:4px solid #2e7d32}.fail{background:#ffebee;border-left:4px solid #c62828}.pending{background:#fff3e0;border-left:4px solid #f57c00}.row{margin:0.5rem 0}.row label{font-weight:600;display:inline-block;width:100px}.match{color:#2e7d32!important;font-weight:600}.mismatch{color:#c62828!important;font-weight:600}textarea{width:100%;height:200px;font-family:monospace;font-size:0.85rem}button{padding:0.5rem 1rem;margin:0.5rem 0.5rem 0.5rem 0;cursor:pointer}button:disabled{opacity:0.5;cursor:not-allowed}.btn-p{background:#0066cc;color:#fff;border:none}.btn-s{background:#f5f5f5;border:1px solid #ddd}footer{margin-top:2rem;padding-top:1rem;border-top:1px solid #ddd;font-size:0.75rem;color:#666}a{color:#0066cc}.nav{margin-bottom:1.5rem}.nav a{margin-right:1rem}table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem}th,td{border:1px solid #ddd;padding:0.5rem;text-align:left}th{background:#f5f5f5}.row-pass{background:#f1f8e9}.row-fail{background:#ffebee}</style>
</head>
<body>
<div class="container">
<div class="banner">
<div style="font-weight:600">MathLedger - v0.2.6</div>
<div><span class="status">LOCKED</span> <a href="/versions/" style="font-size:0.8rem;color:#666">(see /versions/)</a></div>
<div style="font-size:0.8rem;color:#555;margin-top:0.5rem">Tag: <code>v0.2.6-verifier-correctness</code> | Commit: <code>50f323ddd28f</code></div>
</div>
<nav class="nav"><a href="/v0.2.6/">Archive</a> <a href="/v0.2.6/evidence-pack/">Evidence</a> <strong>Verifier</strong> | <a href="/versions/">All</a></nav>
<h1>Evidence Pack Verifier</h1>
<div class="info"><strong>v0.2.6: Cryptographic Parity</strong> - JS verifier now computes byte-for-byte identical hashes to Python. Uses RFC 8785 canonicalization + domain-separated Merkle trees.</div>
<div class="vbox selftest-hero" style="background:#e3f2fd;border-left:4px solid #1976d2;">
<h2 style="margin-top:0;">Run Self-Test Vectors</h2>
<p>Click the button below to run all built-in test vectors. Expected results: valid packs PASS, tampered packs FAIL.</p>
<button class="btn-p btn-large" id="selftest-btn" onclick="runSelfTest()" style="font-size:1.1rem;padding:0.75rem 1.5rem;">Run self-test vectors</button>
<div id="selftest-status" style="margin:0.75rem 0;font-weight:600;font-size:1.1rem;display:none"></div>
<table id="selftest-table" style="display:none;margin-top:1rem;">
<thead><tr><th>Name</th><th>Expected</th><th>Actual</th><th>Test Result</th><th>Reason</th></tr></thead>
<tbody id="selftest-body"></tbody>
</table>
</div>
<div class="vbox">
<h2>Manual Verification</h2>
<textarea id="inp" placeholder="Paste evidence_pack.json..."></textarea>
<div>
<input type="file" id="fi" accept=".json" style="display:none">
<button class="btn-s" onclick="document.getElementById('fi').click()">Upload</button>
<button class="btn-p" onclick="verify()">Verify</button>
</div>
</div>
<div id="res" class="result pending"><strong>Status:</strong> Waiting...</div>
<div id="det" style="display:none">
<h2>Hashes</h2>
<div class="row"><label>U_t:</label> Exp: <code id="eu"></code> Got: <code id="cu"></code></div>
<div class="row"><label>R_t:</label> Exp: <code id="er"></code> Got: <code id="cr"></code></div>
<div class="row"><label>H_t:</label> Exp: <code id="eh"></code> Got: <code id="ch"></code></div>
</div>
<footer>MathLedger v0.2.6 Verifier | <a href="../examples.json">Test Vectors (examples.json)</a></footer>
</div>
<script>
// v0.2.6: Cryptographic Parity with Python
// Domain separation constants (must match Python attestation/dual_root.py)
const DOMAIN_REASONING_LEAF=new Uint8Array([0xA0,...new TextEncoder().encode('reasoning-leaf')]);
const DOMAIN_UI_LEAF=new Uint8Array([0xA1,...new TextEncoder().encode('ui-leaf')]);
const DOMAIN_LEAF=new Uint8Array([0x00]);
const DOMAIN_NODE=new Uint8Array([0x01]);

// RFC 8785 JSON canonicalization
function can(o){if(o===null)return'null';if(typeof o==='boolean')return o?'true':'false';if(typeof o==='number')return Object.is(o,-0)?'0':String(o);if(typeof o==='string'){let r='"';for(let i=0;i<o.length;i++){const c=o.charCodeAt(i);if(c===8)r+='\\b';else if(c===9)r+='\\t';else if(c===10)r+='\\n';else if(c===12)r+='\\f';else if(c===13)r+='\\r';else if(c===34)r+='\\"';else if(c===92)r+='\\\\';else if(c<32)r+='\\u'+c.toString(16).padStart(4,'0');else r+=o[i];}return r+'"';}if(Array.isArray(o))return'['+o.map(can).join(',')+']';if(typeof o==='object'){const k=Object.keys(o).sort();return'{'+k.map(x=>can(x)+':'+can(o[x])).join(',')+'}';}throw Error('bad');}

// SHA256 without domain (for composite root H_t)
async function sha(s){const d=new TextEncoder().encode(s);const h=await crypto.subtle.digest('SHA-256',d);return Array.from(new Uint8Array(h)).map(b=>b.toString(16).padStart(2,'0')).join('');}

// SHA256 with domain prefix (returns hex string)
async function shaD(data,domain){const db=typeof data==='string'?new TextEncoder().encode(data):data;const c=new Uint8Array(domain.length+db.length);c.set(domain);c.set(db,domain.length);const h=await crypto.subtle.digest('SHA-256',c);return Array.from(new Uint8Array(h)).map(b=>b.toString(16).padStart(2,'0')).join('');}

// SHA256 with domain prefix (returns bytes)
async function shaDBytes(data,domain){const db=typeof data==='string'?new TextEncoder().encode(data):data;const c=new Uint8Array(domain.length+db.length);c.set(domain);c.set(db,domain.length);const h=await crypto.subtle.digest('SHA-256',c);return new Uint8Array(h);}

// Merkle root with domain separation (matches Python substrate/crypto/hashing.py)
async function merkleRoot(leafHashes){
if(leafHashes.length===0)return shaD('',DOMAIN_LEAF);
const sorted=[...leafHashes].sort();
let nodes=[];for(const lh of sorted){nodes.push(await shaDBytes(lh,DOMAIN_LEAF));}
while(nodes.length>1){
if(nodes.length%2===1)nodes.push(nodes[nodes.length-1]);
const next=[];
for(let i=0;i<nodes.length;i+=2){
const combined=new Uint8Array(64);combined.set(nodes[i]);combined.set(nodes[i+1],32);
next.push(await shaDBytes(combined,DOMAIN_NODE));}
nodes=next;}
return Array.from(nodes[0]).map(b=>b.toString(16).padStart(2,'0')).join('');}

// Compute U_t from uvil_events
async function computeUt(events){const lh=[];for(const e of events){lh.push(await shaD(can(e),DOMAIN_UI_LEAF));}return merkleRoot(lh);}

// Compute R_t from reasoning_artifacts
async function computeRt(artifacts){const lh=[];for(const a of artifacts){lh.push(await shaD(can(a),DOMAIN_REASONING_LEAF));}return merkleRoot(lh);}

// Compute H_t = SHA256(R_t || U_t)
async function computeHt(rt,ut){return sha(rt+ut);}

document.getElementById('fi').onchange=e=>{if(e.target.files[0]){const r=new FileReader();r.onload=x=>document.getElementById('inp').value=x.target.result;r.readAsText(e.target.files[0]);}};

async function verify(){const R=document.getElementById('res'),D=document.getElementById('det');try{const v=document.getElementById('inp').value.trim();if(!v){R.className='result pending';R.innerHTML='<strong>Status:</strong> No input';D.style.display='none';return;}const p=JSON.parse(v);const uvil=p.uvil_events||[];const arts=p.reasoning_artifacts||[];const eu=p.u_t||'';const er=p.r_t||'';const eh=p.h_t||'';const cu=await computeUt(uvil);const cr=await computeRt(arts);const ch=await computeHt(cr,cu);document.getElementById('eu').textContent=eu||'-';document.getElementById('cu').textContent=cu;document.getElementById('er').textContent=er||'-';document.getElementById('cr').textContent=cr;document.getElementById('eh').textContent=eh||'-';document.getElementById('ch').textContent=ch;const uok=!eu||cu===eu,rok=!er||cr===er,hok=!eh||ch===eh;document.getElementById('cu').className=uok?'match':'mismatch';document.getElementById('cr').className=rok?'match':'mismatch';document.getElementById('ch').className=hok?'match':'mismatch';D.style.display='block';if(!eu&&!er&&!eh){R.className='result pending';R.innerHTML='<strong>Status:</strong> COMPUTED';}else if(uok&&rok&&hok){R.className='result pass';R.innerHTML='<strong>Status:</strong> PASS';}else{R.className='result fail';R.innerHTML='<strong>Status:</strong> FAIL';}}catch(e){R.className='result fail';R.innerHTML='<strong>Status:</strong> '+e.message;D.style.display='none';}}

async function runSelfTest(){const btn=document.getElementById("selftest-btn");const status=document.getElementById("selftest-status");const table=document.getElementById("selftest-table");const tbody=document.getElementById("selftest-body");btn.disabled=true;status.style.display="block";status.textContent="Loading examples.json...";status.className="";tbody.innerHTML="";table.style.display="none";try{const resp=await fetch("../examples.json");if(!resp.ok)throw new Error("examples.json not found");const data=await resp.json();status.textContent="Running tests...";const results=[];const examples=data.examples||{};for(const[name,ex]of Object.entries(examples)){const pack=ex.pack;const expected=ex.expected_verdict||"PASS";const r=await testPack(pack,expected);results.push({name:name,expected:expected,actual:r.actual,pass:r.pass,reason:r.reason});}let allPass=true;for(const r of results){const tr=document.createElement("tr");tr.className=r.pass?"row-pass":"row-fail";const cls=r.pass?"match":"mismatch";const txt=r.pass?"PASS":"FAIL";tr.innerHTML="<td>"+esc(r.name)+"</td><td>"+esc(r.expected)+"</td><td>"+esc(r.actual)+"</td><td class=\""+cls+"\">"+txt+"</td><td>"+esc(r.reason||"-")+"</td>";tbody.appendChild(tr);if(!r.pass)allPass=false;}table.style.display="table";status.className=allPass?"match":"mismatch";status.textContent=allPass?"SELF-TEST PASSED ("+results.length+" vectors)":"SELF-TEST FAILED";}catch(e){status.className="mismatch";status.textContent="Error: "+e.message;}btn.disabled=false;}
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}

// testPack - SEMANTICS: FAIL/FAIL -> test PASSES, PASS/PASS -> test PASSES
async function testPack(pack,expectedResult){try{const uvil=pack.uvil_events||[];const arts=pack.reasoning_artifacts||[];const declaredU=pack.u_t||"";const declaredR=pack.r_t||"";const declaredH=pack.h_t||"";for(const a of arts){if(!("validation_outcome"in a))return{actual:"FAIL",pass:expectedResult==="FAIL",reason:"missing_required_field"};}const computedU=await computeUt(uvil);const computedR=await computeRt(arts);const computedH=await computeHt(computedR,computedU);if(computedU!==declaredU)return{actual:"FAIL",pass:expectedResult==="FAIL",reason:"u_t_mismatch"};if(computedR!==declaredR)return{actual:"FAIL",pass:expectedResult==="FAIL",reason:"r_t_mismatch"};if(computedH!==declaredH)return{actual:"FAIL",pass:expectedResult==="FAIL",reason:"h_t_mismatch"};return{actual:"PASS",pass:expectedResult==="PASS",reason:null};}catch(e){return{actual:"FAIL",pass:expectedResult==="FAIL",reason:e.message};}}
</script>
</body>
</html>'''


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(HTML_TEMPLATE, encoding="utf-8")
    print(f"OK: Written v0.2.6 verifier to {OUTPUT_PATH}")
    print(f"    Size: {len(HTML_TEMPLATE):,} bytes")


if __name__ == "__main__":
    main()

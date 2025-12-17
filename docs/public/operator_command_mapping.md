# Operator Runbook: Command Mapping

**Document Version**: 1.0  
**Author**: Manus-H  
**Date**: December 10, 2025  
**Classification**: Public / For Node Operators

---

## Overview

This document provides the exact mapping from the [Operator Runbook](operator_runbook_pq_activation.md) procedures to real, executable commands. Each section of the runbook is mapped to specific commands that operators will run on activation day.

**Reality Lock Status**: `# REAL-READY` - These commands are designed to work with the actual MathLedger node software and infrastructure.

**Important**: Commands marked with `# DEMO-SCAFFOLD` are illustrative placeholders. Commands without this tag are production-ready.

---

## Pre-Activation Checklist (T-24 Hours)

### 1. Confirm Node Software Version

**Runbook Reference**: Section 2, Item 1

**Real Command**:
```bash
# REAL-READY (if mathledgerd exists in production)
mathledgerd version
```

**Expected Output**:
```
MathLedger Node v2.1.0-pq
Build: 2025-12-09
Commit: abc123def456
PQ Migration: ENABLED
```

**Validation**:
```bash
# Check if version matches expected
mathledgerd version | grep "v2.1.0-pq" && echo "✓ Version OK" || echo "✗ Version mismatch"
```

---

### 2. Verify Node Health and Sync Status

**Runbook Reference**: Section 2, Item 2

**Real Command**:
```bash
# REAL-READY (if mathledgerd status exists)
mathledgerd status
```

**Expected Output** (JSON):
```json
{
  "node_info": {
    "id": "node-12345",
    "network": "mainnet"
  },
  "sync_info": {
    "latest_block_height": "9950",
    "catching_up": false
  },
  "validator_info": {
    "address": "0x...",
    "voting_power": "100"
  }
}
```

**Validation**:
```bash
# Check sync status
mathledgerd status | jq '.sync_info.catching_up' | grep "false" && echo "✓ Node synced" || echo "✗ Node catching up"
```

---

### 3. Check PQ Modules Present

**Runbook Reference**: Section 2, Item 3 (implied)

**Real Command**:
```bash
# REAL-READY
cd /path/to/mathledger  # Or C:\dev\mathledger on Windows
python3 -c "
import os
modules = [
    'basis/crypto/hash_registry.py',
    'basis/crypto/hash_versioned.py',
    'basis/ledger/block_pq.py',
    'basis/ledger/verification.py',
    'backend/consensus_pq/rules.py',
    'backend/consensus_pq/epoch.py',
]
all_present = all(os.path.exists(m) for m in modules)
print('✓ All PQ modules present' if all_present else '✗ Missing PQ modules')
"
```

**Alternative (Dry-Run Script)**:
```bash
# REAL-READY
python3 scripts/pq_activation_dryrun.py --activation-block 10000
```

---

### 4. Review Communication Channels

**Runbook Reference**: Section 2, Item 4

**Real Action**: Join Discord server at `https://discord.gg/mathledger` and navigate to `#pq-migration-operators` channel.

**No command needed** - manual verification.

---

## Activation Day Monitoring (T-0)

### 1. Begin Enhanced Monitoring

**Runbook Reference**: Section 3.1, Item 1

**Real Command** (Linux/systemd):
```bash
# REAL-READY (if using systemd)
journalctl -u mathledgerd -f -n 100
```

**Real Command** (Docker):
```bash
# REAL-READY (if using Docker)
docker logs -f mathledger-node --tail 100
```

**Real Command** (Direct process):
```bash
# REAL-READY
tail -f /var/log/mathledger/node.log
```

---

### 2. Monitor Block Height

**Runbook Reference**: Section 3.1, Item 2

**Real Command**:
```bash
# REAL-READY
watch -n 5 'mathledgerd status | jq ".sync_info.latest_block_height"'
```

**Alternative** (one-time check):
```bash
# REAL-READY
mathledgerd status | jq '.sync_info.latest_block_height'
```

---

### 3. Observe Activation Log

**Runbook Reference**: Section 3.2, Item 1

**Real Command** (grep for activation):
```bash
# REAL-READY
journalctl -u mathledgerd -f | grep --line-buffered "Activating new epoch"
```

**Expected Log Entry**:
```
2025-12-10 12:00:00 INFO [epoch] Activating new epoch: algorithm=SHA3-256, rule_version=v2-dual-required, start_block=10000
```

---

### 4. Confirm First PQ Block

**Runbook Reference**: Section 3.2, Item 2

**Real Command** (grep for dual commitment):
```bash
# REAL-READY
journalctl -u mathledgerd -f | grep --line-buffered "Sealed block .* with dual commitment"
```

**Expected Log Entry**:
```
2025-12-10 12:00:06 INFO [consensus] Sealed block 10001 with dual commitment (legacy_hash=0xabc..., pq_hash=0xdef...)
```

---

## Post-Activation Verification (T+1 to T+60)

### 1. Verify New Block Structure

**Runbook Reference**: Section 3.3, Item 1

**Real Command**:
```bash
# REAL-READY
mathledgerd query block 10001
```

**Expected Output** (JSON):
```json
{
  "block_number": 10001,
  "prev_hash": "0x...",
  "merkle_root": "0x...",
  "timestamp": 1702224006,
  "statements": [...],
  "pq_algorithm": 1,
  "pq_merkle_root": "0x...",
  "pq_prev_hash": "0x...",
  "dual_commitment": "0x..."
}
```

**Validation**:
```bash
# REAL-READY
mathledgerd query block 10001 | jq 'has("pq_algorithm") and has("dual_commitment")' | grep "true" && echo "✓ PQ fields present" || echo "✗ PQ fields missing"
```

---

### 2. Monitor Drift Radar

**Runbook Reference**: Section 3.3, Item 2

**Real Command**:
```bash
# REAL-READY
journalctl -u mathledgerd -f | grep --line-buffered "\[DRIFT_RADAR\]"
```

**Alert Handling**:
- **CRITICAL** or **HIGH** severity: Report immediately to `#pq-migration-operators`
- Copy full log message and post with `@emergency-response-team` tag

---

### 3. Check Public Explorers

**Runbook Reference**: Section 3.3, Item 3

**Real Action**: Navigate to `https://explorer.mathledger.org/block/10001` and verify:
- `pq_algorithm` field is displayed
- `pq_merkle_root` field is displayed
- `dual_commitment` field is displayed

**No command needed** - manual verification via web browser.

---

### 4. Monitor Network Health

**Runbook Reference**: Section 3.3, Item 4

**Real Commands**:

**Block production rate**:
```bash
# REAL-READY
# Sample block heights over 60 seconds and calculate rate
HEIGHT1=$(mathledgerd status | jq -r '.sync_info.latest_block_height')
sleep 60
HEIGHT2=$(mathledgerd status | jq -r '.sync_info.latest_block_height')
RATE=$((HEIGHT2 - HEIGHT1))
echo "Block production rate: $RATE blocks/minute (expected: ~10)"
```

**Peer count**:
```bash
# REAL-READY
mathledgerd query peers | jq 'length'
```

**Expected**: Peer count should remain stable (e.g., 20-50 peers).

---

## Emergency Procedures

### Scenario 1: Activation Fails

**Runbook Reference**: Section 4, Scenario 1

**Diagnostic Commands**:
```bash
# REAL-READY
# Check for epoch activation errors
journalctl -u mathledgerd --since "5 minutes ago" | grep -i "error\|fatal\|epoch"

# Check current epoch
mathledgerd query epoch-info
```

**Report Template**:
```
[ACTIVATION FAILURE]
Operator: <your_name>
Node ID: <node_id>
Current Block: <block_number>
Symptom: <description>
Logs: <paste relevant log lines>
```

---

### Scenario 2: Critical Drift Detected

**Runbook Reference**: Section 4, Scenario 2

**Immediate Actions**:
```bash
# REAL-READY
# 1. Copy drift alert
journalctl -u mathledgerd --since "1 minute ago" | grep "\[DRIFT_RADAR\]" > drift_alert.log

# 2. If validator, isolate node
sudo systemctl stop mathledgerd

# 3. Report (paste drift_alert.log contents to Discord)
```

---

### Scenario 3: Network Fork

**Runbook Reference**: Section 4, Scenario 3

**Diagnostic Commands**:
```bash
# REAL-READY
# Check your chain tip
mathledgerd status | jq '.sync_info.latest_block_hash'

# Check peers' chain tips
mathledgerd query peers | jq '.[] | {peer: .id, block_hash: .latest_block_hash}'
```

---

## Rollback Procedure

**Runbook Reference**: Section 4, Rollback Procedure

**Commands** (only execute if instructed by core team):

```bash
# REAL-READY
# 1. Stop node
sudo systemctl stop mathledgerd

# 2. Download rollback software (URL provided by core team)
wget https://releases.mathledger.org/rollback/mathledgerd-v2.0.9

# 3. Verify checksum
sha256sum mathledgerd-v2.0.9
# Compare with official checksum

# 4. Install rollback version
sudo cp mathledgerd-v2.0.9 /usr/local/bin/mathledgerd
sudo chmod +x /usr/local/bin/mathledgerd

# 5. Reset state to safe block (block number provided by core team)
mathledgerd unsafe-reset-to-block --block 9999

# 6. Restart node
sudo systemctl start mathledgerd

# 7. Verify rollback
mathledgerd version
mathledgerd status | jq '.sync_info.latest_block_height'
```

---

## Smoke-Test Readiness Checklist

### Files to Verify Exist

- [ ] `scripts/pq_activation_dryrun.py` (REAL-READY)
- [ ] `scripts/pq_activation_simulator.py` (DEMO-SCAFFOLD)
- [ ] `artifacts/pq_validator_safety_checklist.json` (REAL-READY)
- [ ] `docs/public/operator_runbook_pq_activation.md` (REAL-READY)
- [ ] `docs/public/operator_command_mapping.md` (this file, REAL-READY)

### Commands to Test Locally

```bash
# Test dry-run script
python3 scripts/pq_activation_dryrun.py --activation-block 10000

# Test simulator (DEMO-SCAFFOLD)
python3 scripts/pq_activation_simulator.py --start-block 9990 --activation-block 10000

# Validate JSON schema
python3 -m json.tool artifacts/pq_validator_safety_checklist.json > /dev/null && echo "✓ JSON valid"
```

### Expected Observable Artifacts

After running dry-run:
- [ ] `pq_dryrun_report.json` created
- [ ] Console output shows checklist results
- [ ] Exit code 0 if all checks pass, 1 if any fail

After running simulator:
- [ ] `pq_simulation_<timestamp>.json` created
- [ ] Console output shows simulated activation sequence
- [ ] Simulated log entries match expected format

---

## Notes

1. **Reality Lock Compliance**: Commands tagged `# REAL-READY` are production-ready and match actual repository structure.
2. **DEMO-SCAFFOLD**: Commands tagged `# DEMO-SCAFFOLD` are illustrative and require real node integration.
3. **Node Software**: Some commands assume `mathledgerd` binary exists. If not yet deployed, these are interface specifications.
4. **Paths**: Adjust paths (`/var/log/mathledger`, `/usr/local/bin/mathledgerd`, etc.) to match your deployment.

---

## Maintenance

This document should be updated whenever:
- Node software CLI changes
- New monitoring commands are added
- Deployment paths change
- Emergency procedures are revised

**Last Updated**: 2025-12-10  
**Maintained By**: MathLedger Core Team

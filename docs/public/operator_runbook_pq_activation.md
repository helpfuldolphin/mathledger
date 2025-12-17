# Operator Runbook: PQ Activation Day

**Document Version**: 1.0  
**Author**: Manus-H, PQ Migration General  
**Date**: December 10, 2025  
**Classification**: Public / For Node Operators

---

## 1. Overview

This runbook provides a step-by-step guide for MathLedger node operators on the day of the post-quantum (PQ) epoch activation. The activation is an automated, on-chain event triggered at a specific block number. Your role as an operator is to monitor the health of your node, verify the successful activation, and be prepared to act in case of unexpected issues.

**Activation Block Number**: `TBD` (To be determined by the passed governance proposal)

**Primary Communication Channel**: `#pq-migration-operators` on Discord (link to be provided).

---

## 2. Pre-Activation Checklist (T-24 Hours)

Complete these steps at least 24 hours before the activation block to ensure you are ready.

-   **[ ] 1. Confirm Node Software Version**: Ensure your node is running the correct software version that includes the PQ migration logic. Use the following command to verify:
    ```bash
    mathledgerd version
    ```
    The version must match the official release announcement (e.g., `v2.1.0-pq`).

-   **[ ] 2. Verify Node Health and Sync Status**: Your node must be fully synced and healthy. Check for any errors in your logs and ensure CPU, memory, and disk usage are within normal ranges.
    ```bash
    mathledgerd status
    ```
    The `catching_up` field should be `false`.

-   **[ ] 3. Check Monitoring and Alerting**: Verify that your monitoring dashboards and alerting systems are operational. You should be monitoring:
    -   Node health (CPU, memory, disk I/O, network I/O)
    -   Block height and sync status
    -   Peer count
    -   Log output for errors or critical warnings

-   **[ ] 4. Review Communication Channels**: Join the `#pq-migration-operators` Discord channel and confirm you are receiving messages. This will be the primary channel for real-time updates and coordination on activation day.

---

## 3. Activation Day Procedure (T-0)

### 3.1. Monitoring Phase (T-60 Minutes to Activation)

-   **[ ] 1. Begin Enhanced Monitoring**: Start closely monitoring your node's health and log output. Keep a terminal open with a live view of your logs:
    ```bash
    journalctl -u mathledgerd -f -n 100
    ```

-   **[ ] 2. Monitor Block Height**: Track the current block height as it approaches the activation block. You can use a block explorer or your node's RPC API.
    ```bash
    mathledgerd status | grep "latest_block_height"
    ```

-   **[ ] 3. Stand By in Communication Channel**: Be present and responsive in the `#pq-migration-operators` Discord channel. The core team will provide status updates as the activation block approaches.

### 3.2. The Activation Event (Block `N`)

This is the moment the `epoch_start_block` is reached. The transition is automatic.

-   **[ ] 1. Observe Activation Log**: At the activation block, you should see a log entry indicating the epoch transition. Look for a message similar to this:
    ```log
    INFO [epoch] Activating new epoch: algorithm=SHA3-256, rule_version=v2-dual-required, start_block=10000
    ```

-   **[ ] 2. Confirm First PQ Block**: The very next block (`N+1`) should be the first block sealed with a dual commitment. Look for a log entry indicating a dual-commitment seal:
    ```log
    INFO [consensus] Sealed block 10001 with dual commitment (legacy_hash=..., pq_hash=...)
    ```

### 3.3. Post-Activation Verification (T+1 to T+60 Minutes)

After the activation block, your primary task is to verify that the chain is stable and producing valid dual-commitment blocks.

-   **[ ] 1. Verify New Block Structure**: Inspect the headers of the new blocks (`N+1`, `N+2`, etc.) using your node's RPC API or a block explorer. Confirm that they contain the new PQ fields:
    -   `pq_algorithm`
    -   `pq_merkle_root`
    -   `dual_commitment`
    ```bash
    mathledgerd query block 10001
    ```

-   **[ ] 2. Monitor Drift Radar**: The node software includes a built-in **PQ Drift Radar**. Monitor your logs for any drift alerts, which will be prefixed with `[DRIFT_RADAR]`.
    ```log
    CRITICAL [DRIFT_RADAR] Dual commitment mismatch detected in block 10002!
    ```
    **Any drift alert is a critical event. Report it immediately.**

-   **[ ] 3. Check Public Explorers**: Verify that public block explorers are correctly interpreting the new blocks and displaying the dual-commitment information.

-   **[ ] 4. Monitor Network Health**: Keep an eye on the overall network health. Look for:
    -   A stable block production rate.
    -   No significant increase in orphaned blocks.
    -   Stable peer counts.

---

## 4. Contingency Plans (Emergency Procedures)

If you observe any of the following issues, follow the steps outlined and **report immediately** in the `#pq-migration-operators` channel.

### Scenario 1: Activation Fails

-   **Symptom**: The activation block is passed, but new blocks are not being produced, or they are still legacy blocks (without PQ fields).
-   **Action**:
    1.  Check your logs for any errors related to epoch activation or block sealing.
    2.  Report your findings immediately with log snippets.
    3.  **Do not restart your node unless instructed.** Await guidance from the core team.

### Scenario 2: Critical Drift Detected

-   **Symptom**: You see a `CRITICAL` or `HIGH` severity alert from the `[DRIFT_RADAR]` in your logs.
-   **Action**:
    1.  Immediately copy the full log message.
    2.  Post it in the `#pq-migration-operators` channel with the tag `@emergency-response-team`.
    3.  Isolate your node from the network to prevent propagating potentially invalid blocks if you are a validator. You can do this by stopping your `mathledgerd` service.
    4.  Await instructions. Do not bring your node back online until the issue is resolved.

### Scenario 3: Network Fork or Instability

-   **Symptom**: You observe a high rate of orphaned blocks, your node frequently switches between forks, or public explorers show conflicting chain tips.
-   **Action**:
    1.  Report the instability, providing as much detail as possible (e.g., conflicting block hashes, peer logs).
    2.  Check your node's view of the chain tip and compare it with other operators in the channel.
    3.  The core team will investigate and declare a canonical chain if necessary. Follow their instructions to ensure you are on the correct fork.

---

## 5. Rollback Procedure

A network-wide rollback is a last resort and will only be initiated by the core team in the event of a catastrophic failure. If a rollback is called:

1.  You will be provided with a specific software version to downgrade to.
2.  You will be given instructions to reset your node's state to a specific, safe block height before the activation.
3.  **Do not attempt to roll back on your own.** This must be a coordinated effort to maintain consensus.

---

## 6. Communication Protocol

-   **Stay Informed**: Keep the `#pq-migration-operators` channel open on a dedicated screen.
-   **Be Clear and Concise**: When reporting an issue, provide your operator name, the symptom, the block height, and relevant log snippets.
-   **Follow Instructions**: The core team and emergency response team will provide authoritative instructions. Please follow them to ensure a coordinated response.

Thank you for your diligence and cooperation in making this historic migration a success.

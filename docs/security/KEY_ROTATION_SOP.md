# GitHub App Key Rotation Standard Operating Procedure (SOP)

| Document ID | Version | Author                  |
|-------------|---------|-------------------------|
| ML-SEC-SOP-001 | 1.0     | Manus K - The Keymaster |

**Status**: `ACTIVE`  
**Date**: `2025-10-31`

---

## 1. Overview

This document outlines the Standard Operating Procedure (SOP) for the rotation of GitHub App private keys used by the MathLedger factory. The primary objective is to mitigate the risk of key compromise by ensuring that private keys are periodically rotated and that a verifiable, sealed log of all rotation events is maintained.

This SOP applies to all agents and automated systems that utilize the MathLedger GitHub App for repository access.

## 2. Rotation Cadence

Private keys for the MathLedger GitHub App **MUST** be rotated every **90 days**.

Emergency rotation **MUST** be performed immediately if a key compromise is suspected or confirmed.

## 3. Rotation Procedure

The rotation process involves generating a new key, updating all relevant systems, recording the event in a sealed log, and revoking the old key.

### 3.1. Generate New Key

1.  Navigate to the MathLedger GitHub App settings page: `https://github.com/organizations/helpfuldolphin/settings/apps/mathledger`
2.  Under the "Private keys" section, click the "Generate a new private key" button.
3.  The new key will be generated and downloaded as a `.pem` file (e.g., `mathledger.2025-10-31.private-key.pem`).

### 3.2. Secure Storage

The newly generated `.pem` file **MUST NOT** be committed to any Git repository. It must be stored in a secure, access-controlled vault or secret management system. For agent-based operations, the key should be provided securely at runtime.

### 3.3. Update Installation & Deploy

1.  Update all relevant agents, CI/CD environments, and local development setups with the content of the new private key.
2.  If a new installation of the App is created, the **Installation ID** must also be updated.
3.  Verify that all systems can successfully authenticate with the new key and installation ID before proceeding.

### 3.4. Record Rotation Event

A new entry **MUST** be appended to the `artifacts/keys/rotation_log.jsonl` file. This is a JSONL file where each line is a sealed, canonical JSON object representing a rotation event.

1.  **Calculate Fingerprints**: Compute the SHA256 hash of the old and new `.pem` files.
    ```bash
    sha256sum <key-file.pem>
    ```
2.  **Construct Event Object**: Create a JSON object with the structure defined in Section 4.
3.  **Generate Seal**: Compute the SHA256 hash of the canonicalized JSON object (RFC8785) to create the `seal`.
4.  **Append to Log**: Append the complete, sealed JSON object as a new line to `rotation_log.jsonl`.

### 3.5. Revoke Old Key

After successfully deploying the new key and recording the rotation event, the old private key **MUST** be revoked.

1.  Return to the GitHub App settings page.
2.  Locate the old key in the "Private keys" list.
3.  Click the "Delete" button next to the old key.

## 4. Sealed Log Format (`rotation_log.jsonl`)

Each line in the log file is a self-contained JSON object, canonicalized according to **RFC8785** before sealing. This ensures that the log is tamper-evident.

| Field                     | Type   | Description                                                                 |
|---------------------------|--------|-----------------------------------------------------------------------------|
| `event_id`                | String | A unique identifier for the event (e.g., UUIDv4).                             |
| `timestamp`               | String | The ISO 8601 UTC timestamp of the rotation event.                           |
| `event_type`              | String | Must be `"KEY_ROTATION"`.                                                   |
| `app_id`                  | String | The GitHub App ID (e.g., `"2144752"`).                                     |
| `installation_id`         | String | The GitHub App Installation ID (e.g., `"92514594"`).                       |
| `new_key_fingerprint`     | String | The SHA256 hash of the new private key PEM file.                            |
| `revoked_key_fingerprint` | String | The SHA256 hash of the old private key PEM file. (Can be `null` for first key). |
| `rotated_by`              | String | The name of the agent or user performing the rotation (e.g., `"Manus K"`).     |
| `seal`                    | String | The SHA256 hash of the canonicalized JSON object (excluding the `seal` field). |

### Example Event:

```json
{"app_id":"2144752","event_id":"...","event_type":"KEY_ROTATION","installation_id":"92514594","new_key_fingerprint":"nLPU9...","revoked_key_fingerprint":"...","rotated_by":"Manus K","seal":"...","timestamp":"2025-10-31T18:30:00Z"}
```

## 5. Emergency Rotation

In the event of a suspected key compromise, the following steps must be taken immediately:

1.  **Revoke Compromised Key**: Immediately revoke the suspected key via the GitHub App settings.
2.  **Generate New Key**: Generate a new private key.
3.  **Deploy New Key**: Update all systems with the new key.
4.  **Audit**: Investigate the scope of the compromise by analyzing repository access logs.
5.  **Record Event**: Log the emergency rotation in `rotation_log.jsonl`, noting the reason in the commit message.

---

### References

[1] GitHub Docs: "Managing private keys for GitHub Apps" - [https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/managing-private-keys-for-github-apps](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/managing-private-keys-for-github-apps)

[2] RFC 8785: "JSON Canonicalization Scheme (JCS)" - [https://www.rfc-editor.org/rfc/rfc8785](https://www.rfc-editor.org/rfc/rfc8785)


# Scenario Suite Results

- Suite version: `v1`
- Generated at (UTC): `2026-02-23T21:20:08+00:00`
- Total scenarios: `9`
- Total steps: `11`
- Matches: `11/11`
- Mismatches: `0`
- CEO fraud progression: `YELLOW -> RED -> RED` (expected `YELLOW -> RED -> RED`)
- CAC dual-gate RED: `True`
- Benign all GREEN: `True`

| Scenario | Step | Type | Expected | Actual | Risk | Convergence | Match | H_t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adv-ciso-firewall-exception | request_1 | adversarial | YELLOW | YELLOW | YELLOW (exp YELLOW) | GREEN (exp GREEN) | True | `bddef3e81cb4...` |
| adv-emergency-ssh-credentials | request_1 | adversarial | RED | RED | RED (exp RED) | RED (exp RED) | True | `3a8a77913ba3...` |
| adv-fight-flight-soc | request_1 | adversarial | RED | RED | RED (exp RED) | RED (exp RED) | True | `0785ddfee6f4...` |
| adv-ceo-fraud-sequence | request_1_1030 | adversarial | YELLOW | YELLOW | YELLOW (exp YELLOW) | GREEN (exp GREEN) | True | `d8819c0f0782...` |
| adv-ceo-fraud-sequence | request_2_1100 | adversarial | RED | RED | RED (exp RED) | YELLOW (exp YELLOW) | True | `beb5e86b1bc1...` |
| adv-ceo-fraud-sequence | request_3_1115 | adversarial | RED | RED | YELLOW (exp YELLOW) | RED (exp RED) | True | `0068b7100562...` |
| adv-command-authority-confusion | request_1 | adversarial | RED | RED | RED (exp RED) | RED (exp RED) | True | `3f50fbf0c305...` |
| ctrl-standard-access-ticket | request_1 | benign | GREEN | GREEN | GREEN (exp GREEN) | GREEN (exp GREEN) | True | `70f7658bbe71...` |
| ctrl-planned-maintenance-window | request_1 | benign | GREEN | GREEN | GREEN (exp GREEN) | GREEN (exp GREEN) | True | `aeb0476c0d73...` |
| ctrl-routine-exec-request | request_1 | benign | GREEN | GREEN | GREEN (exp GREEN) | GREEN (exp GREEN) | True | `5e65245bdf54...` |
| ctrl-incident-escalation-policy | request_1 | benign | GREEN | GREEN | GREEN (exp GREEN) | GREEN (exp GREEN) | True | `04f8db13cc11...` |

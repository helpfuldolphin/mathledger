# Wave A0 + A1 Claim Packet

## Claim A0

Claim ID: `ML-A0-BOOTSTRAP-DETERMINISTIC`  
Tier: A  
Statement: Project bootstrap is deterministic and executable from clean checkout.

Falsifier:
- test runner cannot import package from `src`
- bootstrap requires hidden environment state

Closure artifact targets:
- `pyproject.toml`
- `README.md` quickstart
- deterministic test run (`pytest -q`)

## Claim A1

Claim ID: `ML-A1-BASIS-GENOME`  
Tier: A  
Statement: Basis primitives are pure and deterministic for fixed inputs.

Falsifier:
- inconsistent hashing/roots across repeated runs
- non-canonical serialization drift
- any primitive reads wall clock, random source, or external IO

Closure artifact targets:
- basis module implementation under `src/mathledger/basis`
- deterministic tests in `tests/test_wave_a1_basis_core.py`

## Lane Discipline in A0/A1

- No authority promotion logic yet.
- No Lane B integration yet.
- No TDA/USLA gating in this wave.


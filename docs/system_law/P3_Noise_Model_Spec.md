# P3 Noise Model Specification

> **Status**: DESIGN-ONLY Specification
> **Phase**: X P3 (SHADOW MODE ONLY)
> **Version**: 1.0.0
> **Date**: 2025-12-10

---

## Table of Contents

1. [Overview](#1-overview)
2. [Allowed Noise Distributions](#2-allowed-noise-distributions)
3. [Required Parameter Bounds](#3-required-parameter-bounds)
4. [Synthetic Pathologies Specification](#4-synthetic-pathologies-specification)
5. [Synthetic State Snapshot Schemas](#5-synthetic-state-snapshot-schemas)
6. [Validation Requirements](#6-validation-requirements)
7. [Summary](#7-summary)

---

## 1. Overview

This document formalizes the noise model specification for Phase X P3 First-Light shadow experiments. All noise injection operates in SHADOW MODE: observation and logging only, with no governance modification.

### 1.1 Design Principles

| Principle | Description |
|-----------|-------------|
| Deterministic Reproducibility | All noise draws derive from a seeded PRNG hierarchy |
| Bounded Pathology | Synthetic failures stay within specified bounds |
| Observable Only | Noise decisions logged, never enforced |
| Schema Compliance | All snapshots conform to versioned JSON schemas |

### 1.2 Noise Model Categories

The P3 noise model comprises seven regimes:

1. **Base Noise** — Independent Bernoulli events (timeout, spurious fail/pass)
2. **Correlated Failures** — Latent factor model with spatial correlation
3. **Cluster Degradation** — Hidden Markov Model (HEALTHY/DEGRADED states)
4. **Heat-Death** — Resource depletion process
5. **Heavy-Tailed Timeouts** — Mixture distribution (Exponential + Pareto)
6. **Non-Stationary Noise** — Time-varying drift
7. **Adaptive Noise** — Policy-aware confidence scaling

---

## 2. Allowed Noise Distributions

### 2.1 Bernoulli Distribution

**Use Case**: Base noise events (timeout, spurious failure, spurious pass)

**Specification**:
```
X ~ Bernoulli(p)
P(X = 1) = p
P(X = 0) = 1 - p
```

**Parameters**:
| Parameter | Type | Bounds | Description |
|-----------|------|--------|-------------|
| `p` | float | [0.0, 1.0] | Event probability |

**Sampling**:
```python
def sample_bernoulli(prng: DeterministicPRNG, p: float) -> bool:
    return prng.random() < p
```

---

### 2.2 Gaussian (Normal) Distribution

**Use Case**: Resource consumption/recovery, noise perturbations, drift modeling

**Specification**:
```
X ~ N(mu, sigma^2)
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))
```

**Parameters**:
| Parameter | Type | Bounds | Description |
|-----------|------|--------|-------------|
| `mu` | float | (-inf, +inf) | Mean |
| `sigma` | float | [0.0, +inf) | Standard deviation |

**Sampling**:
```python
def sample_gaussian(prng: DeterministicPRNG, mu: float, sigma: float) -> float:
    return prng.gauss(mu, sigma)
```

**Truncation** (when required):
```python
def sample_truncated_gaussian(prng, mu, sigma, low, high, max_attempts=1000):
    for _ in range(max_attempts):
        x = prng.gauss(mu, sigma)
        if low <= x <= high:
            return x
    return max(low, min(high, mu))  # Fallback to clamped mean
```

---

### 2.3 Exponential Distribution

**Use Case**: Fast timeout durations, inter-arrival times

**Specification**:
```
X ~ Exp(lambda)
f(x) = lambda * exp(-lambda * x)  for x >= 0
E[X] = 1 / lambda
```

**Parameters**:
| Parameter | Type | Bounds | Description |
|-----------|------|--------|-------------|
| `lambda` | float | (0.0, +inf) | Rate parameter |

**Sampling**:
```python
def sample_exponential(prng: DeterministicPRNG, rate: float) -> float:
    return prng.expovariate(rate)
```

---

### 2.4 Pareto Distribution

**Use Case**: Heavy-tailed timeout durations (extreme events)

**Specification**:
```
X ~ Pareto(alpha, x_min)
f(x) = (alpha * x_min^alpha) / x^(alpha + 1)  for x >= x_min
E[X] = (alpha * x_min) / (alpha - 1)  for alpha > 1
```

**Parameters**:
| Parameter | Type | Bounds | Description |
|-----------|------|--------|-------------|
| `alpha` | float | (0.0, +inf) | Tail index (shape) |
| `x_min` | float | (0.0, +inf) | Scale parameter (minimum value) |

**Sampling**:
```python
def sample_pareto(prng: DeterministicPRNG, alpha: float, x_min: float) -> float:
    u = prng.random()
    return x_min / (u ** (1.0 / alpha))
```

**Note**: Lower `alpha` produces heavier tails. Recommended: `alpha >= 1.0` for finite mean.

---

### 2.5 Mixture Distribution (Exponential + Pareto)

**Use Case**: Heavy-tailed timeout modeling with common fast events and rare extreme events

**Specification**:
```
X ~ (1 - pi) * Exp(lambda) + pi * Pareto(alpha, x_min)

With probability (1 - pi): X ~ Exp(lambda)
With probability pi:       X ~ Pareto(alpha, x_min)
```

**Parameters**:
| Parameter | Type | Bounds | Description |
|-----------|------|--------|-------------|
| `pi` | float | [0.0, 1.0] | Mixing probability (Pareto weight) |
| `lambda` | float | (0.0, +inf) | Exponential rate |
| `alpha` | float | (1.0, +inf) | Pareto tail index |
| `x_min` | float | (0.0, +inf) | Pareto scale |

**Sampling**:
```python
def sample_mixture(prng, pi, lambda_exp, alpha, x_min):
    if prng.random() < pi:
        return sample_pareto(prng, alpha, x_min)
    else:
        return sample_exponential(prng, lambda_exp)
```

---

### 2.6 Hidden Markov Model (Two-State)

**Use Case**: Cluster degradation (HEALTHY <-> DEGRADED state transitions)

**Specification**:
```
States: S = {HEALTHY, DEGRADED}

Transition Matrix:
           HEALTHY    DEGRADED
HEALTHY    1-alpha    alpha
DEGRADED   beta       1-beta

Emission (failure probability):
P(fail | HEALTHY)  = theta_healthy
P(fail | DEGRADED) = theta_degraded
```

**Parameters**:
| Parameter | Type | Bounds | Description |
|-----------|------|--------|-------------|
| `alpha` | float | [0.0, 1.0] | P(HEALTHY -> DEGRADED) |
| `beta` | float | [0.0, 1.0] | P(DEGRADED -> HEALTHY) |
| `theta_healthy` | float | [0.0, 1.0] | Failure rate in HEALTHY |
| `theta_degraded` | float | [0.0, 1.0] | Failure rate in DEGRADED |

**State Transition**:
```python
def transition(prng, current_state, alpha, beta):
    if current_state == HEALTHY:
        return DEGRADED if prng.random() < alpha else HEALTHY
    else:
        return HEALTHY if prng.random() < beta else DEGRADED
```

---

### 2.7 Linear Drift Model

**Use Case**: Non-stationary noise with time-varying parameters

**Specification**:
```
theta(t) = theta_0 + delta * t + epsilon(t)
epsilon(t) ~ N(0, sigma^2)

Clamped to [0, 1] for probability parameters.
```

**Parameters**:
| Parameter | Type | Bounds | Description |
|-----------|------|--------|-------------|
| `theta_0` | float | [0.0, 1.0] | Initial noise rate |
| `delta` | float | [-0.01, 0.01] | Drift rate per cycle |
| `sigma` | float | [0.0, 0.1] | Noise standard deviation |

**Sampling**:
```python
def get_drifted_rate(prng, cycle, theta_0, delta, sigma):
    drift = theta_0 + delta * cycle
    noise = prng.gauss(0.0, sigma)
    return max(0.0, min(1.0, drift + noise))
```

---

## 3. Required Parameter Bounds

### 3.1 Base Noise Parameters

| Parameter | Symbol | Min | Max | Default | Units |
|-----------|--------|-----|-----|---------|-------|
| Timeout rate | `p_timeout` | 0.0 | 0.30 | 0.05 | probability |
| Spurious fail rate | `p_fail` | 0.0 | 0.15 | 0.02 | probability |
| Spurious pass rate | `p_pass` | 0.0 | 0.10 | 0.01 | probability |

**Invariant**: `p_timeout + p_fail + p_pass <= 0.40`

---

### 3.2 Correlated Failure Parameters

| Parameter | Symbol | Min | Max | Default | Units |
|-----------|--------|-----|-----|---------|-------|
| Factor activation prob | `rho` | 0.0 | 0.30 | 0.10 | probability |
| Per-factor failure rate | `theta_k` | 0.0 | 0.50 | 0.20 | probability |
| Max factors per item | `max_factors` | 1 | 5 | 3 | count |

**Invariant**: `rho * max(theta_k) <= 0.15` (bounds correlation impact)

---

### 3.3 Cluster Degradation Parameters

| Parameter | Symbol | Min | Max | Default | Units |
|-----------|--------|-----|-----|---------|-------|
| Healthy -> Degraded | `alpha` | 0.0 | 0.15 | 0.05 | probability |
| Degraded -> Healthy | `beta` | 0.05 | 0.50 | 0.20 | probability |
| Healthy failure rate | `theta_h` | 0.0 | 0.05 | 0.01 | probability |
| Degraded failure rate | `theta_d` | 0.10 | 0.50 | 0.30 | probability |

**Invariants**:
- `beta > alpha` (ensures recovery faster than degradation)
- `theta_d > theta_h` (degraded state worse than healthy)
- Stationary distribution: `P(HEALTHY) = beta / (alpha + beta) >= 0.50`

---

### 3.4 Heat-Death Parameters

| Parameter | Symbol | Min | Max | Default | Units |
|-----------|--------|-----|-----|---------|-------|
| Initial resource | `R_0` | 500 | 2000 | 1000 | units |
| Minimum resource | `R_min` | 50 | 200 | 100 | units |
| Consumption mean | `mu_c` | 1.0 | 20.0 | 10.0 | units/cycle |
| Consumption std | `sigma_c` | 0.0 | 10.0 | 5.0 | units/cycle |
| Recovery mean | `mu_r` | 1.0 | 20.0 | 8.0 | units/cycle |
| Recovery std | `sigma_r` | 0.0 | 10.0 | 3.0 | units/cycle |

**Invariants**:
- `mu_r >= 0.7 * mu_c` (ensures eventual recovery possible)
- `R_0 >= 5 * R_min` (sufficient initial headroom)
- Expected drift: `E[R(t+1) - R(t)] = mu_r - mu_c` should be >= -2.0

---

### 3.5 Heavy-Tail Parameters

| Parameter | Symbol | Min | Max | Default | Units |
|-----------|--------|-----|-----|---------|-------|
| Pareto mixing prob | `pi` | 0.0 | 0.20 | 0.10 | probability |
| Exponential rate | `lambda` | 0.01 | 1.0 | 0.10 | 1/ms |
| Pareto tail index | `alpha` | 1.1 | 3.0 | 1.5 | dimensionless |
| Pareto scale | `x_min` | 50 | 500 | 100 | ms |

**Invariants**:
- `alpha > 1.0` (finite mean required)
- `pi <= 0.20` (extreme events remain rare)
- Expected timeout: `E[T] <= 1000 ms`

---

### 3.6 Non-Stationary Parameters

| Parameter | Symbol | Min | Max | Default | Units |
|-----------|--------|-----|-----|---------|-------|
| Initial rate | `theta_0` | 0.0 | 0.30 | 0.10 | probability |
| Drift rate | `delta` | -0.001 | 0.001 | 0.0001 | prob/cycle |
| Noise std | `sigma` | 0.0 | 0.05 | 0.01 | probability |

**Invariants**:
- `|delta| * 1000 <= 0.50` (drift bounded over 1000 cycles)
- `theta_0 + delta * T_max + 3*sigma <= 1.0` (stays valid probability)

---

### 3.7 Adaptive Noise Parameters

| Parameter | Symbol | Min | Max | Default | Units |
|-----------|--------|-----|-----|---------|-------|
| Adaptation strength | `gamma` | 0.0 | 1.0 | 0.50 | dimensionless |

**Model**: `theta_adaptive = theta_base * (1 + gamma * confidence)`
where `confidence = |policy_prob - 0.5| * 2`

**Invariant**: `theta_base * (1 + gamma) <= 0.50` (capped noise rate)

---

## 4. Synthetic Pathologies Specification

Synthetic pathologies are controlled noise injections for testing system robustness. Each pathology has defined characteristics, trigger conditions, and severity levels.

### 4.1 Spike Pathology

**Description**: Sudden, transient increase in failure rate.

**Specification**:
```
spike(t, t_start, duration, magnitude):
    if t_start <= t < t_start + duration:
        return base_rate + magnitude
    else:
        return base_rate
```

**Parameters**:
| Parameter | Type | Bounds | Description |
|-----------|------|--------|-------------|
| `t_start` | int | [0, T_max] | Spike onset cycle |
| `duration` | int | [1, 20] | Spike duration in cycles |
| `magnitude` | float | [0.05, 0.40] | Added failure rate |

**Severity Levels**:
| Level | Duration | Magnitude | Total Impact |
|-------|----------|-----------|--------------|
| MILD | 1-5 | 0.05-0.10 | < 0.50 |
| MODERATE | 5-10 | 0.10-0.20 | 0.50-2.0 |
| SEVERE | 10-20 | 0.20-0.40 | 2.0-8.0 |

**Schema**:
```json
{
  "pathology_type": "spike",
  "t_start": 100,
  "duration": 10,
  "magnitude": 0.15,
  "severity": "MODERATE",
  "affected_items": ["*"],
  "description": "Transient failure spike at cycle 100"
}
```

---

### 4.2 Drift Pathology

**Description**: Gradual, monotonic change in failure rate over time.

**Specification**:
```
drift(t, t_start, rate, direction):
    if t >= t_start:
        delta = rate * (t - t_start) * direction
        return clamp(base_rate + delta, 0.0, 1.0)
    else:
        return base_rate
```

**Parameters**:
| Parameter | Type | Bounds | Description |
|-----------|------|--------|-------------|
| `t_start` | int | [0, T_max] | Drift onset cycle |
| `rate` | float | [0.0001, 0.005] | Drift rate per cycle |
| `direction` | int | {-1, +1} | Drift direction |

**Severity Levels**:
| Level | Rate | 500-Cycle Impact |
|-------|------|------------------|
| MILD | 0.0001-0.0005 | 0.05-0.25 |
| MODERATE | 0.0005-0.002 | 0.25-1.0 |
| SEVERE | 0.002-0.005 | 1.0-2.5 |

**Schema**:
```json
{
  "pathology_type": "drift",
  "t_start": 50,
  "rate": 0.001,
  "direction": 1,
  "severity": "MODERATE",
  "max_rate": 0.50,
  "description": "Gradual degradation starting cycle 50"
}
```

---

### 4.3 Oscillation Pathology

**Description**: Periodic variation in failure rate.

**Specification**:
```
oscillation(t, period, amplitude, phase):
    return base_rate + amplitude * sin(2 * pi * t / period + phase)
```

**Parameters**:
| Parameter | Type | Bounds | Description |
|-----------|------|--------|-------------|
| `period` | int | [10, 200] | Oscillation period in cycles |
| `amplitude` | float | [0.02, 0.15] | Peak deviation from base |
| `phase` | float | [0, 2*pi] | Initial phase |

**Severity Levels**:
| Level | Amplitude | Period |
|-------|-----------|--------|
| MILD | 0.02-0.05 | 50-200 |
| MODERATE | 0.05-0.10 | 20-50 |
| SEVERE | 0.10-0.15 | 10-20 |

**Schema**:
```json
{
  "pathology_type": "oscillation",
  "period": 50,
  "amplitude": 0.08,
  "phase": 0.0,
  "severity": "MODERATE",
  "description": "Periodic instability with 50-cycle period"
}
```

---

### 4.4 Cluster Burst Pathology

**Description**: Correlated failures affecting groups of related items.

**Specification**:
```
cluster_burst(t, t_start, duration, affected_clusters, burst_rate):
    if t_start <= t < t_start + duration:
        for item in affected_clusters:
            P(fail | item) = burst_rate
    else:
        normal failure rates
```

**Parameters**:
| Parameter | Type | Bounds | Description |
|-----------|------|--------|-------------|
| `t_start` | int | [0, T_max] | Burst onset |
| `duration` | int | [5, 50] | Burst duration |
| `cluster_fraction` | float | [0.1, 0.5] | Fraction of items affected |
| `burst_rate` | float | [0.2, 0.8] | Failure rate during burst |

**Schema**:
```json
{
  "pathology_type": "cluster_burst",
  "t_start": 200,
  "duration": 25,
  "cluster_fraction": 0.3,
  "burst_rate": 0.5,
  "severity": "SEVERE",
  "affected_factors": ["tactic_ring", "tactic_simp"],
  "description": "Correlated cluster failure at cycle 200"
}
```

---

### 4.5 Heat-Death Cascade Pathology

**Description**: Resource exhaustion leading to system-wide failures.

**Specification**:
```
heat_death_cascade(t, depletion_rate):
    R(t) = R_0 * exp(-depletion_rate * t)
    if R(t) < R_critical:
        P(fail) = 1.0 - (R(t) / R_critical)
```

**Parameters**:
| Parameter | Type | Bounds | Description |
|-----------|------|--------|-------------|
| `depletion_rate` | float | [0.001, 0.02] | Exponential decay rate |
| `R_critical` | float | [50, 200] | Critical resource threshold |

**Schema**:
```json
{
  "pathology_type": "heat_death_cascade",
  "depletion_rate": 0.005,
  "R_critical": 100,
  "expected_onset_cycle": 460,
  "severity": "SEVERE",
  "description": "Resource exhaustion cascade"
}
```

---

### 4.6 Combined Pathology Profiles

**MILD Profile** (Testing):
```json
{
  "profile_name": "mild_stress",
  "pathologies": [
    {"type": "spike", "t_start": 100, "duration": 3, "magnitude": 0.05},
    {"type": "drift", "t_start": 200, "rate": 0.0002, "direction": 1}
  ],
  "expected_impact": "minimal",
  "use_case": "Regression testing"
}
```

**MODERATE Profile** (Shadow Testing):
```json
{
  "profile_name": "moderate_stress",
  "pathologies": [
    {"type": "spike", "t_start": 150, "duration": 8, "magnitude": 0.12},
    {"type": "oscillation", "period": 40, "amplitude": 0.06}
  ],
  "expected_impact": "observable",
  "use_case": "Shadow experiment validation"
}
```

**SEVERE Profile** (Boundary Testing):
```json
{
  "profile_name": "severe_stress",
  "pathologies": [
    {"type": "cluster_burst", "t_start": 100, "duration": 30, "burst_rate": 0.5},
    {"type": "drift", "t_start": 0, "rate": 0.003, "direction": 1},
    {"type": "heat_death_cascade", "depletion_rate": 0.008}
  ],
  "expected_impact": "system boundary violation",
  "use_case": "Red-flag observation testing"
}
```

---

## 5. Synthetic State Snapshot Schemas

### 5.1 Noise Model State Snapshot

**Schema Version**: `p3-noise-state/1.0.0`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "P3NoiseModelStateSnapshot",
  "type": "object",
  "required": ["schema_version", "cycle", "timestamp", "mode", "base_state", "regime_states"],
  "properties": {
    "schema_version": {
      "const": "p3-noise-state/1.0.0"
    },
    "cycle": {
      "type": "integer",
      "minimum": 0
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "mode": {
      "const": "SHADOW"
    },
    "master_seed": {
      "type": "integer"
    },
    "base_state": {
      "$ref": "#/definitions/BaseNoiseState"
    },
    "regime_states": {
      "$ref": "#/definitions/RegimeStates"
    },
    "active_pathologies": {
      "type": "array",
      "items": {"$ref": "#/definitions/PathologyState"}
    },
    "decisions_this_cycle": {
      "$ref": "#/definitions/CycleDecisions"
    }
  },
  "definitions": {
    "BaseNoiseState": {
      "type": "object",
      "properties": {
        "timeout_rate": {"type": "number", "minimum": 0, "maximum": 1},
        "spurious_fail_rate": {"type": "number", "minimum": 0, "maximum": 1},
        "spurious_pass_rate": {"type": "number", "minimum": 0, "maximum": 1}
      }
    },
    "RegimeStates": {
      "type": "object",
      "properties": {
        "correlated": {
          "type": "object",
          "properties": {
            "enabled": {"type": "boolean"},
            "active_factors": {"type": "object"},
            "rho": {"type": "number"}
          }
        },
        "degradation": {
          "type": "object",
          "properties": {
            "enabled": {"type": "boolean"},
            "state": {"enum": ["HEALTHY", "DEGRADED"]},
            "alpha": {"type": "number"},
            "beta": {"type": "number"}
          }
        },
        "heat_death": {
          "type": "object",
          "properties": {
            "enabled": {"type": "boolean"},
            "resource_level": {"type": "number"},
            "R_min": {"type": "number"}
          }
        },
        "heavy_tail": {
          "type": "object",
          "properties": {
            "enabled": {"type": "boolean"},
            "pi": {"type": "number"},
            "lambda": {"type": "number"},
            "alpha": {"type": "number"},
            "x_min": {"type": "number"}
          }
        },
        "nonstationary": {
          "type": "object",
          "properties": {
            "enabled": {"type": "boolean"},
            "current_rate": {"type": "number"},
            "theta_0": {"type": "number"},
            "delta": {"type": "number"}
          }
        },
        "adaptive": {
          "type": "object",
          "properties": {
            "enabled": {"type": "boolean"},
            "gamma": {"type": "number"}
          }
        }
      }
    },
    "PathologyState": {
      "type": "object",
      "properties": {
        "pathology_type": {"type": "string"},
        "active": {"type": "boolean"},
        "remaining_cycles": {"type": "integer"},
        "current_impact": {"type": "number"}
      }
    },
    "CycleDecisions": {
      "type": "object",
      "properties": {
        "total_items": {"type": "integer"},
        "timeout_count": {"type": "integer"},
        "spurious_fail_count": {"type": "integer"},
        "spurious_pass_count": {"type": "integer"},
        "clean_count": {"type": "integer"}
      }
    }
  }
}
```

### 5.2 Example State Snapshot

```json
{
  "schema_version": "p3-noise-state/1.0.0",
  "cycle": 142,
  "timestamp": "2025-12-10T14:30:00.000000+00:00",
  "mode": "SHADOW",
  "master_seed": 12345,
  "base_state": {
    "timeout_rate": 0.05,
    "spurious_fail_rate": 0.02,
    "spurious_pass_rate": 0.01
  },
  "regime_states": {
    "correlated": {
      "enabled": true,
      "active_factors": {
        "tactic_ring": true,
        "tactic_simp": false,
        "tactic_omega": false
      },
      "rho": 0.10
    },
    "degradation": {
      "enabled": true,
      "state": "HEALTHY",
      "alpha": 0.05,
      "beta": 0.20
    },
    "heat_death": {
      "enabled": true,
      "resource_level": 847.3,
      "R_min": 100.0
    },
    "heavy_tail": {
      "enabled": true,
      "pi": 0.10,
      "lambda": 0.10,
      "alpha": 1.5,
      "x_min": 100.0
    },
    "nonstationary": {
      "enabled": true,
      "current_rate": 0.1142,
      "theta_0": 0.10,
      "delta": 0.0001
    },
    "adaptive": {
      "enabled": true,
      "gamma": 0.50
    }
  },
  "active_pathologies": [
    {
      "pathology_type": "drift",
      "active": true,
      "remaining_cycles": null,
      "current_impact": 0.0142
    }
  ],
  "decisions_this_cycle": {
    "total_items": 50,
    "timeout_count": 3,
    "spurious_fail_count": 1,
    "spurious_pass_count": 0,
    "clean_count": 46
  }
}
```

---

### 5.3 Pathology Configuration Schema

**Schema Version**: `p3-pathology-config/1.0.0`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "P3PathologyConfiguration",
  "type": "object",
  "required": ["schema_version", "profile_name", "pathologies"],
  "properties": {
    "schema_version": {
      "const": "p3-pathology-config/1.0.0"
    },
    "profile_name": {
      "type": "string",
      "pattern": "^[a-z_]+$"
    },
    "description": {
      "type": "string"
    },
    "pathologies": {
      "type": "array",
      "items": {
        "oneOf": [
          {"$ref": "#/definitions/SpikePathology"},
          {"$ref": "#/definitions/DriftPathology"},
          {"$ref": "#/definitions/OscillationPathology"},
          {"$ref": "#/definitions/ClusterBurstPathology"},
          {"$ref": "#/definitions/HeatDeathPathology"}
        ]
      }
    },
    "expected_impact": {
      "enum": ["minimal", "observable", "significant", "system boundary violation"]
    }
  },
  "definitions": {
    "SpikePathology": {
      "type": "object",
      "required": ["type", "t_start", "duration", "magnitude"],
      "properties": {
        "type": {"const": "spike"},
        "t_start": {"type": "integer", "minimum": 0},
        "duration": {"type": "integer", "minimum": 1, "maximum": 20},
        "magnitude": {"type": "number", "minimum": 0.05, "maximum": 0.40},
        "severity": {"enum": ["MILD", "MODERATE", "SEVERE"]}
      }
    },
    "DriftPathology": {
      "type": "object",
      "required": ["type", "t_start", "rate", "direction"],
      "properties": {
        "type": {"const": "drift"},
        "t_start": {"type": "integer", "minimum": 0},
        "rate": {"type": "number", "minimum": 0.0001, "maximum": 0.005},
        "direction": {"enum": [-1, 1]},
        "max_rate": {"type": "number", "maximum": 0.50}
      }
    },
    "OscillationPathology": {
      "type": "object",
      "required": ["type", "period", "amplitude"],
      "properties": {
        "type": {"const": "oscillation"},
        "period": {"type": "integer", "minimum": 10, "maximum": 200},
        "amplitude": {"type": "number", "minimum": 0.02, "maximum": 0.15},
        "phase": {"type": "number", "minimum": 0, "maximum": 6.283185}
      }
    },
    "ClusterBurstPathology": {
      "type": "object",
      "required": ["type", "t_start", "duration", "cluster_fraction", "burst_rate"],
      "properties": {
        "type": {"const": "cluster_burst"},
        "t_start": {"type": "integer", "minimum": 0},
        "duration": {"type": "integer", "minimum": 5, "maximum": 50},
        "cluster_fraction": {"type": "number", "minimum": 0.1, "maximum": 0.5},
        "burst_rate": {"type": "number", "minimum": 0.2, "maximum": 0.8},
        "affected_factors": {"type": "array", "items": {"type": "string"}}
      }
    },
    "HeatDeathPathology": {
      "type": "object",
      "required": ["type", "depletion_rate"],
      "properties": {
        "type": {"const": "heat_death_cascade"},
        "depletion_rate": {"type": "number", "minimum": 0.001, "maximum": 0.02},
        "R_critical": {"type": "number", "minimum": 50, "maximum": 200}
      }
    }
  }
}
```

---

### 5.4 Noise Decision Log Schema

**Schema Version**: `p3-noise-decision/1.0.0`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "P3NoiseDecisionLog",
  "type": "object",
  "required": ["schema_version", "cycle", "item", "decision", "contributing_factors"],
  "properties": {
    "schema_version": {
      "const": "p3-noise-decision/1.0.0"
    },
    "cycle": {
      "type": "integer"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "item": {
      "type": "string"
    },
    "decision": {
      "enum": ["CLEAN", "TIMEOUT", "SPURIOUS_FAIL", "SPURIOUS_PASS"]
    },
    "computed_rates": {
      "type": "object",
      "properties": {
        "timeout_rate": {"type": "number"},
        "spurious_fail_rate": {"type": "number"},
        "spurious_pass_rate": {"type": "number"}
      }
    },
    "contributing_factors": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "regime": {"type": "string"},
          "contribution": {"type": "string"},
          "value": {"type": "number"}
        }
      }
    },
    "prng_path": {
      "type": "string"
    },
    "draw_value": {
      "type": "number"
    }
  }
}
```

---

## 6. Validation Requirements

### 6.1 Parameter Validation

All noise model configurations MUST pass validation before use:

```python
def validate_noise_config(config: UnifiedNoiseConfig) -> List[str]:
    """Validate noise configuration against P3 bounds."""
    errors = []

    # Base noise bounds
    if config.base_noise.timeout_rate > 0.30:
        errors.append("timeout_rate exceeds maximum 0.30")
    if config.base_noise.spurious_fail_rate > 0.15:
        errors.append("spurious_fail_rate exceeds maximum 0.15")
    if config.base_noise.spurious_pass_rate > 0.10:
        errors.append("spurious_pass_rate exceeds maximum 0.10")

    total_base = (config.base_noise.timeout_rate +
                  config.base_noise.spurious_fail_rate +
                  config.base_noise.spurious_pass_rate)
    if total_base > 0.40:
        errors.append(f"Total base noise {total_base:.3f} exceeds 0.40")

    # Correlated noise bounds
    if config.correlated.enabled:
        if config.correlated.rho > 0.30:
            errors.append("correlated.rho exceeds maximum 0.30")
        max_theta = max(config.correlated.theta.values(), default=0)
        if config.correlated.rho * max_theta > 0.15:
            errors.append("Correlated impact rho*max(theta) exceeds 0.15")

    # Degradation bounds
    if config.degradation.enabled:
        if config.degradation.beta <= config.degradation.alpha:
            errors.append("degradation.beta must exceed alpha")
        if config.degradation.theta_degraded <= config.degradation.theta_healthy:
            errors.append("degradation.theta_degraded must exceed theta_healthy")

    # Heat death bounds
    if config.heat_death.enabled:
        if config.heat_death.recovery_mean < 0.7 * config.heat_death.consumption_mean:
            errors.append("Heat death recovery_mean too low relative to consumption")

    # Non-stationary bounds
    if config.nonstationary.enabled:
        max_drift = abs(config.nonstationary.delta) * 1000
        if max_drift > 0.50:
            errors.append(f"Non-stationary drift over 1000 cycles ({max_drift:.3f}) exceeds 0.50")

    return errors
```

### 6.2 Snapshot Validation

All state snapshots MUST be validated against schemas before logging:

```python
def validate_snapshot(snapshot: dict, schema: dict) -> bool:
    """Validate snapshot against JSON schema."""
    import jsonschema
    try:
        jsonschema.validate(snapshot, schema)
        return True
    except jsonschema.ValidationError:
        return False
```

### 6.3 SHADOW Mode Invariants

| Invariant | Check |
|-----------|-------|
| Mode field | `snapshot["mode"] == "SHADOW"` |
| No enforcement | Decisions logged only, never returned for use |
| Deterministic | Same seed + cycle produces identical decisions |
| Bounded impact | Total noise rate <= 0.50 in all cases |

---

## 7. External Verifier Guidance: Interpreting noise_summary in the Evidence Pack

This section provides guidance for external verifiers interpreting the `noise_summary` field in the First Light Evidence Pack. The noise summary is attached at `evidence["governance"]["noise"]` and provides transparency into synthetic noise injection during P3 shadow experiments.

### 7.1 Purpose of noise_summary

The `noise_summary` in the Evidence Pack serves several purposes:

1. **Transparency**: Documents all synthetic noise injection applied during the experiment
2. **Reproducibility**: Enables external verification that noise was within P3 bounds
3. **Impact Assessment**: Quantifies how noise affected key metrics (delta_p, RSI)
4. **SHADOW Mode Verification**: Confirms all noise was observational only

### 7.2 Key Fields for External Verification

| Field | Purpose | Verification Check |
|-------|---------|-------------------|
| `schema_version` | Schema compliance | Should be `p3-noise-summary/1.0.0` |
| `mode` | SHADOW mode confirmation | MUST be `"SHADOW"` |
| `regime_proportions` | Noise regime activation rates | All values in [0.0, 1.0] |
| `delta_p_aggregate.total_contribution` | Net delta_p impact | Typically negative (noise hurts learning) |
| `delta_p_aggregate.magnitude_class` | Impact severity | LOW, MODERATE, or HIGH |
| `rsi_aggregate.suppression_class` | RSI suppression level | LOW, MODERATE, or HIGH |

### 7.3 Interpreting Regime Proportions

The `regime_proportions` field shows the fraction of cycles where each noise regime was active:

```json
{
  "base": 1.0,                    // Always active
  "correlated": 0.15,             // 15% of cycles had correlated factor activation
  "degradation_healthy": 0.72,    // 72% of cycles in HEALTHY state
  "degradation_degraded": 0.28,   // 28% of cycles in DEGRADED state
  "heat_death_nominal": 0.85,     // 85% of cycles with normal resources
  "heat_death_stressed": 0.15,    // 15% of cycles with stressed resources
  "heavy_tail": 1.0,              // Heavy-tail enabled throughout
  "nonstationary": 1.0,           // Non-stationary enabled throughout
  "adaptive": 1.0,                // Adaptive enabled throughout
  "pathology": 0.10               // 10% of cycles had pathology injection
}
```

**Verification guidance**:
- `degradation_degraded` > 0.50 indicates excessive time in degraded state
- `heat_death_stressed` > 0.30 indicates resource exhaustion stress test
- `pathology` > 0.20 indicates heavy synthetic stress injection

### 7.4 Interpreting Delta-p Aggregate

The `delta_p_aggregate` field quantifies noise impact on the learning trajectory:

```json
{
  "total_contribution": -0.45,     // Net delta_p impact (negative = hurt learning)
  "avg_per_cycle": -0.0045,        // Per-cycle average
  "by_noise_type": {
    "timeout": -0.30,              // Timeouts contributed -0.30 to delta_p
    "spurious_fail": -0.15,        // Spurious failures contributed -0.15
    "spurious_pass": 0.0           // Spurious passes had no net contribution
  },
  "net_direction": "NEGATIVE",     // Overall direction
  "magnitude_class": "MODERATE"    // Impact severity
}
```

**Verification guidance**:
- `magnitude_class == "HIGH"` indicates significant noise impact requiring careful interpretation of delta_p results
- `net_direction == "NEGATIVE"` is expected (noise typically hurts learning)
- Compare `total_contribution` to observed delta_p to assess noise fraction

### 7.5 Interpreting RSI Aggregate

The `rsi_aggregate` field quantifies noise impact on stability (RSI):

```json
{
  "noise_event_rate": 0.08,              // 8% of decisions were noise events
  "estimated_rsi_suppression": 0.008,    // Estimated 0.8% RSI suppression
  "suppression_class": "LOW",            // Suppression severity
  "degraded_cycle_fraction": 0.20,       // 20% of cycles in degraded state
  "pathology_active_fraction": 0.10      // 10% of cycles had pathology
}
```

**Verification guidance**:
- `suppression_class == "HIGH"` indicates RSI trajectory significantly affected by noise
- `degraded_cycle_fraction > 0.30` indicates prolonged degradation stress
- When interpreting RSI results, account for `estimated_rsi_suppression`

### 7.6 Governance Advisories

The Evidence Pack may include governance advisories related to noise impact:

| Advisory Type | Severity | Meaning |
|--------------|----------|---------|
| `NOISE_IMPACT` | INFO | Moderate noise impact observed |
| `NOISE_IMPACT` | WARN | High noise impact - interpret results carefully |
| `RSI_SUPPRESSION` | WARN | RSI significantly suppressed by noise |

### 7.7 SHADOW Mode Verification Checklist

External verifiers should confirm:

- [ ] `mode == "SHADOW"` in noise_summary
- [ ] All regime proportions are in [0.0, 1.0]
- [ ] No `enforcement` or `active_blocking` fields present
- [ ] Noise parameters are within P3 bounds (see Section 3)
- [ ] Advisories are INFO or WARN severity only (no CRITICAL/BLOCK)

### 7.8 Example: Interpreting a Complete noise_summary

```json
{
  "schema_version": "p3-noise-summary/1.0.0",
  "mode": "SHADOW",
  "total_cycles": 1000,
  "regime_proportions": {
    "base": 1.0,
    "degradation_degraded": 0.18,
    "pathology": 0.05
  },
  "delta_p_aggregate": {
    "total_contribution": -0.35,
    "magnitude_class": "MODERATE"
  },
  "rsi_aggregate": {
    "suppression_class": "LOW"
  },
  "interpretation_guidance": "Moderate delta_p impact from noise injection."
}
```

**Interpretation**: This experiment had moderate noise impact (-0.35 total delta_p contribution) with low RSI suppression. The degradation regime was active 18% of the time, and pathology injection occurred in 5% of cycles. Results should be interpreted with awareness that ~0.35 units of delta_p degradation came from synthetic noise rather than system behavior.

---

## 8. When Real Telemetry Arrives (P5 Transition)

This section clarifies the boundary between P3 synthetic noise modeling and P5 real telemetry analysis.

### 8.1 P3 Noise Model Scope

The P3 noise model is **exclusively for synthetic simulator behavior**. It provides:

- **Controlled noise injection** with known parameters and deterministic PRNG
- **Bounded synthetic pathologies** (spike, drift, oscillation, cluster burst, heat-death)
- **Observational shadow mode** for testing system robustness

**Critical distinction**: P3 noise parameters (timeout_rate, spurious_fail_rate, degradation alpha/beta, etc.) are synthetic knobs that model *hypothetical* failure modes. They are **not** calibrated to real-world behavior.

### 8.2 P5 Real Telemetry: Different Interpretation Framework

When P5 real telemetry becomes available, it should **not** be interpreted through P3 noise knobs:

| Aspect | P3 Synthetic Noise | P5 Real Telemetry |
|--------|-------------------|-------------------|
| **Source** | Controlled PRNG injection | Actual USLA system behavior |
| **Parameters** | Configured noise rates | Observed divergence patterns |
| **Pathologies** | Synthetic (known onset/duration) | Emergent (detected, not injected) |
| **Interpretation** | "Did the model handle this noise?" | "What is the real system doing?" |
| **Calibration** | P3 bounds (Section 3) | Empirical from production data |

**Do not** attempt to map P5 observations back to P3 noise parameters. P5 divergence patterns may reflect:
- Real infrastructure variance (not modeled in P3)
- Emergent system behavior (not synthetic pathology)
- Production load characteristics (not bounded noise)

### 8.3 Visual Comparison: P3 noise_summary vs P5 divergence_summary

To compare P3 synthetic behavior against P5 real telemetry in a single figure:

**Recommended Figure Layout**: Side-by-side dual-panel time series with shared x-axis (cycle number).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    P3 vs P5 Divergence Comparison                       │
├─────────────────────────────────┬───────────────────────────────────────┤
│  Panel A: P3 Synthetic Noise    │  Panel B: P5 Real Divergence          │
│                                 │                                       │
│  ───────────────────────────    │  ───────────────────────────────      │
│  [Stacked area: regime          │  [Line plot: twin delta_p vs          │
│   proportions over time]        │   real delta_p over time]             │
│                                 │                                       │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─     │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─       │
│  [Bar: delta_p contribution     │  [Scatter: divergence magnitude       │
│   by noise type]                │   at each cycle]                      │
│                                 │                                       │
│  Annotations:                   │  Annotations:                         │
│  • Pathology injection markers  │  • Red-flag events detected           │
│  • Regime transition points     │  • Anomaly clusters                   │
│                                 │                                       │
├─────────────────────────────────┴───────────────────────────────────────┤
│  Panel C: Overlay Comparison (optional)                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  [Dual y-axis: P3 noise_event_rate (left) vs P5 divergence_rate (right)]│
│  [Highlight: regions where P5 exceeds P3 synthetic bounds]              │
└─────────────────────────────────────────────────────────────────────────┘
```

**Panel A (P3 Synthetic)**:
- Top: Stacked area chart showing regime activation over cycles (base, correlated, degradation, etc.)
- Bottom: Horizontal bar showing cumulative delta_p contribution by noise type
- Vertical markers for pathology injection events

**Panel B (P5 Real)**:
- Top: Dual line plot comparing twin model delta_p vs real system delta_p
- Bottom: Scatter plot showing divergence magnitude at each observation cycle
- Red markers for detected red-flag events or anomalies

**Panel C (Overlay, optional)**:
- Dual y-axis time series comparing P3 `noise_event_rate` against P5 `divergence_rate`
- Shaded regions where P5 divergence exceeds P3 synthetic bounds (indicating real behavior more extreme than synthetic stress tests)
- Useful for identifying whether P3 noise model adequately brackets real-world variance

**Key visual cues**:
- Use distinct color palettes: cool tones (blue/green) for P3 synthetic, warm tones (orange/red) for P5 real
- Shared x-axis enables direct temporal alignment
- Include legend distinguishing "injected" (P3) vs "observed" (P5) events

### 8.4 Transition Guidance

When P5 real telemetry is available:

1. **Do not recalibrate P3 noise parameters** to match P5 observations
2. **Do** use P5 data to validate whether P3 stress profiles were realistic
3. **Do** create separate P5 divergence analysis using `divergence_summary` schema
4. **Do** compare P3 vs P5 visually to assess synthetic model coverage
5. **Consider** whether P5 reveals failure modes not represented in P3 pathologies

### 8.5 Noise vs Reality Dashboard Specification

This section specifies the 2-panel dashboard for comparing P3 synthetic noise against P5 real divergence.

#### 8.5.1 Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│             NOISE VS REALITY DASHBOARD — P3/P5 Comparison                   │
│  Experiment: {experiment_id}    Generated: {timestamp}    Mode: SHADOW      │
├────────────────────────────────────┬────────────────────────────────────────┤
│                                    │                                        │
│  ┌──────────────────────────────┐  │  ┌──────────────────────────────────┐  │
│  │   PANEL A: P3 SYNTHETIC      │  │  │   PANEL B: P5 REAL DIVERGENCE    │  │
│  │   Noise Regime Proportions   │  │  │   Δp Scatter + Red Flags         │  │
│  │                              │  │  │                                  │  │
│  │  100%├────────────────────┐  │  │  │  Δp │    ∘  ∘                    │  │
│  │      │████████████████████│  │  │  │ 0.02├──∘──∘──●──∘──∘────────     │  │
│  │   75%│████████████████    │  │  │  │     │  ∘     ●                   │  │
│  │      │████████████        │  │  │  │ 0.01├──∘──∘──∘──∘──∘──∘──        │  │
│  │   50%│████████            │  │  │  │     │  ∘  ∘  ∘  ∘  ∘             │  │
│  │      │████                │  │  │  │    0├──∘──∘──∘──∘──∘──∘──        │  │
│  │   25%│                    │  │  │  │     │                            │  │
│  │      │                    │  │  │  │-0.01├──────────────────────      │  │
│  │    0%└────────────────────┘  │  │  │     └────────────────────────    │  │
│  │       0   100  200  300  400 │  │  │       0   100  200  300  400     │  │
│  │              Cycle           │  │  │              Cycle               │  │
│  │                              │  │  │                                  │  │
│  │  Legend:                     │  │  │  Legend:                         │  │
│  │  ■ base        ■ correlated  │  │  │  ∘ twin_delta_p                  │  │
│  │  ■ degradation ■ heat_death  │  │  │  ● red_flag_event                │  │
│  │  ■ heavy_tail  ■ pathology   │  │  │  ─ real_delta_p (reference)      │  │
│  │                              │  │  │                                  │  │
│  └──────────────────────────────┘  │  └──────────────────────────────────┘  │
│                                    │                                        │
├────────────────────────────────────┴────────────────────────────────────────┤
│  SUMMARY METRICS                                                            │
│  ┌────────────────────────┬────────────────────────┬───────────────────────┐│
│  │ P3 Noise Event Rate    │ P5 Divergence Rate     │ Coverage Ratio        ││
│  │ 8.2%                   │ 5.1%                   │ 1.61 (P3 > P5)        ││
│  └────────────────────────┴────────────────────────┴───────────────────────┘│
│  Advisory: P3 synthetic stress adequately brackets observed P5 divergence.  │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 8.5.2 Panel A: P3 Synthetic Noise Regime Proportions

**Purpose**: Visualize which noise regimes were active during the P3 synthetic experiment.

**Chart Type**: Stacked area chart (100% normalized) over cycle number.

**Visual Elements**:
| Element | Description | Color (suggested) |
|---------|-------------|-------------------|
| base | Base noise layer (always present) | Blue (#3498db) |
| correlated | Correlated factor activation | Green (#2ecc71) |
| degradation | HMM degradation state active | Yellow (#f1c40f) |
| heat_death | Heat-death resource stress | Orange (#e67e22) |
| heavy_tail | Heavy-tail timeout active | Purple (#9b59b6) |
| pathology | Synthetic pathology injection | Red (#e74c3c) |

**Minimal Input Data for Panel A**:

```json
{
  "panel_a_input": {
    "experiment_id": "string",
    "total_cycles": "integer",
    "regime_time_series": [
      {
        "cycle": 0,
        "base": 1.0,
        "correlated": 0.0,
        "degradation": 0.0,
        "heat_death": 0.0,
        "heavy_tail": 0.0,
        "pathology": 0.0
      }
    ],
    "regime_proportions_aggregate": {
      "base": "float [0,1]",
      "correlated": "float [0,1]",
      "degradation_healthy": "float [0,1]",
      "degradation_degraded": "float [0,1]",
      "heat_death_nominal": "float [0,1]",
      "heat_death_stressed": "float [0,1]",
      "heavy_tail": "float [0,1]",
      "pathology": "float [0,1]"
    },
    "pathology_markers": [
      {
        "cycle": "integer",
        "type": "spike|drift|oscillation|cluster_burst|heat_death_cascade",
        "severity": "MILD|MODERATE|SEVERE"
      }
    ],
    "delta_p_by_noise_type": {
      "timeout": "float",
      "spurious_fail": "float",
      "spurious_pass": "float"
    }
  }
}
```

**Required Fields**:
- `total_cycles`: Total experiment duration
- `regime_time_series`: Per-cycle regime activation (for stacked area)
- `regime_proportions_aggregate`: Summary proportions (for tooltip/legend)
- `pathology_markers`: Injection points for annotation markers

#### 8.5.3 Panel B: P5 Real Divergence / Δp Scatter

**Purpose**: Visualize real system divergence between twin model and production telemetry.

**Chart Type**: Scatter plot with optional reference line.

**Visual Elements**:
| Element | Description | Symbol/Style |
|---------|-------------|--------------|
| twin_delta_p | Twin model predicted Δp | Open circle (∘) |
| real_delta_p | Real system observed Δp | Solid line (─) |
| red_flag_event | Detected anomaly/red-flag | Filled circle (●) red |
| divergence_band | ±1σ divergence envelope | Shaded region |

**Minimal Input Data for Panel B**:

```json
{
  "panel_b_input": {
    "experiment_id": "string",
    "total_cycles": "integer",
    "divergence_time_series": [
      {
        "cycle": 0,
        "twin_delta_p": "float",
        "real_delta_p": "float",
        "divergence_magnitude": "float",
        "is_red_flag": "boolean"
      }
    ],
    "divergence_aggregate": {
      "mean_divergence": "float",
      "std_divergence": "float",
      "max_divergence": "float",
      "divergence_rate": "float [0,1]",
      "red_flag_count": "integer",
      "red_flag_cycles": ["integer"]
    },
    "real_telemetry_source": {
      "provider": "string (e.g., 'usla_adapter')",
      "start_timestamp": "ISO8601",
      "end_timestamp": "ISO8601"
    }
  }
}
```

**Required Fields**:
- `divergence_time_series`: Per-cycle twin vs real comparison
- `divergence_aggregate`: Summary statistics for dashboard metrics
- `real_telemetry_source`: Provenance for P5 data

#### 8.5.4 noise_vs_reality_summary JSON Schema

**Schema Version**: `noise-vs-reality/1.0.0`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "NoiseVsRealitySummary",
  "description": "SPEC-ONLY: Unified summary comparing P3 synthetic noise against P5 real divergence",
  "type": "object",
  "required": [
    "schema_version",
    "experiment_id",
    "generated_at",
    "p3_summary",
    "p5_summary",
    "comparison_metrics",
    "coverage_assessment"
  ],
  "properties": {
    "schema_version": {
      "const": "noise-vs-reality/1.0.0"
    },
    "experiment_id": {
      "type": "string",
      "description": "Unique identifier linking P3 and P5 experiments"
    },
    "generated_at": {
      "type": "string",
      "format": "date-time"
    },
    "mode": {
      "const": "SHADOW",
      "description": "Confirms both P3 and P5 were shadow-only"
    },
    "p3_summary": {
      "type": "object",
      "description": "P3 synthetic noise summary (from build_noise_summary_for_p3)",
      "required": ["total_cycles", "noise_event_rate", "regime_proportions", "delta_p_contribution"],
      "properties": {
        "total_cycles": {
          "type": "integer",
          "minimum": 1
        },
        "noise_event_rate": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "Fraction of decisions that were noise events"
        },
        "regime_proportions": {
          "type": "object",
          "properties": {
            "base": {"type": "number"},
            "correlated": {"type": "number"},
            "degradation_degraded": {"type": "number"},
            "heat_death_stressed": {"type": "number"},
            "pathology": {"type": "number"}
          }
        },
        "delta_p_contribution": {
          "type": "object",
          "properties": {
            "total": {"type": "number"},
            "by_type": {
              "type": "object",
              "properties": {
                "timeout": {"type": "number"},
                "spurious_fail": {"type": "number"},
                "spurious_pass": {"type": "number"}
              }
            }
          }
        },
        "pathology_events": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "cycle": {"type": "integer"},
              "type": {"type": "string"},
              "severity": {"type": "string"}
            }
          }
        }
      }
    },
    "p5_summary": {
      "type": "object",
      "description": "P5 real telemetry divergence summary",
      "required": ["total_cycles", "divergence_rate", "divergence_stats", "red_flags"],
      "properties": {
        "total_cycles": {
          "type": "integer",
          "minimum": 1
        },
        "divergence_rate": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "Fraction of cycles with significant divergence"
        },
        "divergence_stats": {
          "type": "object",
          "properties": {
            "mean": {"type": "number"},
            "std": {"type": "number"},
            "max": {"type": "number"},
            "p95": {"type": "number"}
          }
        },
        "red_flags": {
          "type": "object",
          "properties": {
            "count": {"type": "integer"},
            "types": {
              "type": "object",
              "additionalProperties": {"type": "integer"}
            },
            "cycles": {
              "type": "array",
              "items": {"type": "integer"}
            }
          }
        },
        "telemetry_source": {
          "type": "object",
          "properties": {
            "provider": {"type": "string"},
            "start_timestamp": {"type": "string", "format": "date-time"},
            "end_timestamp": {"type": "string", "format": "date-time"}
          }
        }
      }
    },
    "comparison_metrics": {
      "type": "object",
      "description": "Direct P3 vs P5 comparison metrics",
      "required": ["coverage_ratio", "noise_vs_divergence_rate", "delta_p_correlation"],
      "properties": {
        "coverage_ratio": {
          "type": "number",
          "description": "p3_noise_event_rate / p5_divergence_rate (>1 means P3 brackets P5)"
        },
        "noise_vs_divergence_rate": {
          "type": "object",
          "properties": {
            "p3_noise_rate": {"type": "number"},
            "p5_divergence_rate": {"type": "number"},
            "difference": {"type": "number"}
          }
        },
        "delta_p_correlation": {
          "type": "object",
          "description": "Correlation between P3 noise impact and P5 divergence",
          "properties": {
            "pearson_r": {"type": "number", "minimum": -1, "maximum": 1},
            "interpretation": {"type": "string"}
          }
        },
        "exceedance_cycles": {
          "type": "array",
          "description": "Cycles where P5 divergence exceeded P3 synthetic bounds",
          "items": {"type": "integer"}
        },
        "exceedance_rate": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "Fraction of cycles where P5 exceeded P3 bounds"
        }
      }
    },
    "coverage_assessment": {
      "type": "object",
      "description": "Assessment of whether P3 adequately models P5 behavior",
      "required": ["verdict", "confidence", "gaps"],
      "properties": {
        "verdict": {
          "enum": ["ADEQUATE", "MARGINAL", "INSUFFICIENT"],
          "description": "Overall assessment of P3 coverage"
        },
        "confidence": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "reasoning": {
          "type": "string"
        },
        "gaps": {
          "type": "array",
          "description": "Failure modes observed in P5 but not represented in P3",
          "items": {
            "type": "object",
            "properties": {
              "description": {"type": "string"},
              "p5_frequency": {"type": "number"},
              "suggested_p3_pathology": {"type": "string"}
            }
          }
        },
        "recommendations": {
          "type": "array",
          "items": {"type": "string"}
        }
      }
    },
    "governance_advisory": {
      "type": "object",
      "properties": {
        "severity": {
          "enum": ["INFO", "WARN"]
        },
        "message": {"type": "string"},
        "action_required": {"type": "boolean"}
      }
    }
  }
}
```

#### 8.5.5 Example noise_vs_reality_summary Instance

```json
{
  "schema_version": "noise-vs-reality/1.0.0",
  "experiment_id": "first-light-2025-12-11-001",
  "generated_at": "2025-12-11T10:30:00.000000+00:00",
  "mode": "SHADOW",
  "p3_summary": {
    "total_cycles": 1000,
    "noise_event_rate": 0.082,
    "regime_proportions": {
      "base": 1.0,
      "correlated": 0.15,
      "degradation_degraded": 0.18,
      "heat_death_stressed": 0.05,
      "pathology": 0.08
    },
    "delta_p_contribution": {
      "total": -0.35,
      "by_type": {
        "timeout": -0.22,
        "spurious_fail": -0.13,
        "spurious_pass": 0.0
      }
    },
    "pathology_events": [
      {"cycle": 150, "type": "spike", "severity": "MODERATE"},
      {"cycle": 400, "type": "drift", "severity": "MILD"}
    ]
  },
  "p5_summary": {
    "total_cycles": 1000,
    "divergence_rate": 0.051,
    "divergence_stats": {
      "mean": 0.003,
      "std": 0.008,
      "max": 0.045,
      "p95": 0.018
    },
    "red_flags": {
      "count": 3,
      "types": {"DELTA_P_SPIKE": 2, "RSI_DROP": 1},
      "cycles": [287, 512, 891]
    },
    "telemetry_source": {
      "provider": "usla_adapter",
      "start_timestamp": "2025-12-11T08:00:00.000000+00:00",
      "end_timestamp": "2025-12-11T10:00:00.000000+00:00"
    }
  },
  "comparison_metrics": {
    "coverage_ratio": 1.61,
    "noise_vs_divergence_rate": {
      "p3_noise_rate": 0.082,
      "p5_divergence_rate": 0.051,
      "difference": 0.031
    },
    "delta_p_correlation": {
      "pearson_r": 0.23,
      "interpretation": "Weak positive correlation; P3 noise partially explains P5 variance"
    },
    "exceedance_cycles": [512],
    "exceedance_rate": 0.001
  },
  "coverage_assessment": {
    "verdict": "ADEQUATE",
    "confidence": 0.85,
    "reasoning": "P3 synthetic noise rate (8.2%) exceeds P5 observed divergence rate (5.1%) with coverage ratio 1.61. Only 1 cycle (0.1%) showed P5 divergence exceeding P3 bounds.",
    "gaps": [],
    "recommendations": [
      "Continue monitoring for RSI_DROP patterns not fully modeled in P3",
      "Consider adding cluster correlation for DELTA_P_SPIKE events"
    ]
  },
  "governance_advisory": {
    "severity": "INFO",
    "message": "P3 synthetic stress adequately brackets observed P5 divergence.",
    "action_required": false
  }
}
```

#### 8.5.6 Smoke-Test Readiness Checklist

Before deploying the Noise vs Reality Dashboard, verify:

**Data Pipeline**:
- [ ] P3 `noise_summary` can be extracted from Evidence Pack at `evidence["governance"]["noise"]`
- [ ] P5 `divergence_summary` schema is finalized (see Phase_X_P4_Spec.md)
- [ ] Both summaries share compatible `experiment_id` for linking
- [ ] Time series data is aligned on cycle number (not wall-clock time)

**Panel A (P3 Synthetic)**:
- [ ] `regime_time_series` array is populated with per-cycle regime flags
- [ ] `regime_proportions_aggregate` sums to expected values (base always 1.0)
- [ ] `pathology_markers` array contains all injection events with correct cycle numbers
- [ ] Stacked area chart renders without gaps or overlaps

**Panel B (P5 Real)**:
- [ ] `divergence_time_series` contains `twin_delta_p` and `real_delta_p` for each cycle
- [ ] Red-flag events are marked with `is_red_flag: true`
- [ ] `divergence_aggregate.divergence_rate` is calculated correctly
- [ ] Scatter plot renders with correct symbol differentiation

**Comparison Metrics**:
- [ ] `coverage_ratio` = p3_noise_rate / p5_divergence_rate (handle div-by-zero)
- [ ] `exceedance_cycles` identifies cycles where |p5_divergence| > max(p3_noise_impact)
- [ ] `delta_p_correlation.pearson_r` is computed from aligned time series

**Coverage Assessment**:
- [ ] `verdict` logic: ADEQUATE if coverage_ratio >= 1.0 and exceedance_rate < 0.05
- [ ] `verdict` logic: MARGINAL if coverage_ratio >= 0.8 or exceedance_rate < 0.10
- [ ] `verdict` logic: INSUFFICIENT otherwise
- [ ] `gaps` array populated when P5 shows patterns not in P3 pathology set

**Schema Validation**:
- [ ] Output validates against `noise-vs-reality/1.0.0` JSON schema
- [ ] All timestamps are ISO8601 format
- [ ] All rates are in [0.0, 1.0] range
- [ ] `mode` field is always "SHADOW"

**Governance**:
- [ ] Advisory severity is INFO or WARN only (no CRITICAL in shadow mode)
- [ ] `action_required` is false for ADEQUATE/MARGINAL verdicts
- [ ] Dashboard does not modify any governance state

**Visual Rendering**:
- [ ] Color palette distinguishes P3 (cool: blue/green) from P5 (warm: orange/red)
- [ ] Shared x-axis (cycle) enables temporal alignment
- [ ] Legend clearly labels "injected" vs "observed"
- [ ] Summary metrics bar displays coverage_ratio, noise_rate, divergence_rate

---

## 9. Summary

This P3 Noise Model Specification defines:

1. **Seven allowed distributions**: Bernoulli, Gaussian, Exponential, Pareto, Mixture, HMM, Linear Drift
2. **Bounded parameters**: All noise rates constrained to prevent runaway behavior
3. **Five synthetic pathologies**: Spike, Drift, Oscillation, Cluster Burst, Heat-Death
4. **Three severity levels**: MILD, MODERATE, SEVERE with defined impact ranges
5. **Versioned JSON schemas**: State snapshots, pathology configs, decision logs
6. **Validation requirements**: Parameter bounds, schema compliance, SHADOW mode invariants

**SHADOW MODE CONTRACT MAINTAINED**: All noise decisions are observational only. No governance modifications. No enforcement actions.

**Implementation requires explicit future authorization.**

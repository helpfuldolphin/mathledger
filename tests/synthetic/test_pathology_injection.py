import math

import pytest

from backend.synthetic.pathology_injection import (
    inject_drift,
    inject_oscillation,
    inject_spike,
)

pytestmark = pytest.mark.unit


def test_inject_spike_preserves_shape_and_source():
    baseline = [0.1, 0.2, 0.3]
    result = inject_spike(baseline, at=1, magnitude=5.0)

    assert len(result) == len(baseline)
    assert result == pytest.approx([0.1, 5.2, 0.3])
    assert baseline == [0.1, 0.2, 0.3]


def test_inject_drift_applies_linear_offset():
    baseline = [1.0, 1.0, 1.0, 1.0]
    result = inject_drift(baseline, slope=0.5)

    assert len(result) == len(baseline)
    assert result == pytest.approx([1.0, 1.5, 2.0, 2.5])
    assert baseline == [1.0, 1.0, 1.0, 1.0]


def test_inject_oscillation_adds_sinusoid_without_mutation():
    baseline = [0.0] * 6
    result = inject_oscillation(baseline, period=3, amplitude=2.0)

    expected = [
        0.0,
        2.0 * math.sin(2 * math.pi / 3),
        2.0 * math.sin(4 * math.pi / 3),
        0.0,
        2.0 * math.sin(8 * math.pi / 3),
        2.0 * math.sin(10 * math.pi / 3),
    ]

    assert len(result) == len(baseline)
    assert result == pytest.approx(expected)
    assert baseline == [0.0] * 6

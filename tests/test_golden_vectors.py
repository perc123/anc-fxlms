"""The committed golden vectors must match what the generator produces.

If one of these tests fails, either the Python reference changed behaviour
(a bug, or an intended change that needs `python -m anc.golden --write`), or
the files were edited by hand.
"""

import numpy as np
import pytest

from anc.golden import CASES, FORMAT_VERSION, build, load

CASE_NAMES = [case.name for case in CASES]


@pytest.mark.parametrize("case", CASES, ids=CASE_NAMES)
def test_golden_files_match_generator(case):
    fresh = build(case)
    stored = load(case.name)

    # Everything except the drift-derived tolerances must match exactly.
    tolerance_keys = {"y_tolerance", "w_tolerance"}
    assert {k: v for k, v in stored.params.items() if k not in tolerance_keys} == \
           {k: v for k, v in fresh.params.items() if k not in tolerance_keys}

    # Tolerances come from float32 drift, which can shift slightly between
    # platforms. Same order of magnitude is what matters.
    assert 0.5 <= fresh.y_tolerance / stored.y_tolerance <= 2.0
    assert 0.5 <= fresh.w_tolerance / stored.w_tolerance <= 2.0

    # Not bit-exact: NumPy's sin() and summation may differ in the last bit
    # between NumPy versions and CPUs. 1e-12 is about 10^4 ULPs at 1.0, far
    # below any behavioural change.
    for stored_values, fresh_values in [
        (stored.x, fresh.x), (stored.e, fresh.e),
        (stored.y64, fresh.y64), (stored.w64, fresh.w64),
    ]:
        np.testing.assert_allclose(fresh_values, stored_values, rtol=0, atol=1e-12)

    # The float32 reference is only reproducible to float32 precision.
    assert np.max(np.abs(fresh.y32 - stored.y32)) <= stored.y_tolerance
    assert np.max(np.abs(fresh.w32 - stored.w32)) <= stored.w_tolerance


@pytest.mark.parametrize("name", CASE_NAMES)
def test_golden_files_are_consistent(name):
    data = load(name)
    n = int(data.params["n_samples"])
    length = int(data.params["filter_length"])

    assert int(data.params["format_version"]) == FORMAT_VERSION
    assert data.x.shape == data.e.shape == data.y64.shape == data.y32.shape == (n,)
    assert data.w64.shape == data.w32.shape == (length,)
    assert np.isfinite(data.y64).all() and np.isfinite(data.w64).all()

    # The float32 reference must lie within the stored tolerance of float64
    # (by construction it is within a third of it).
    assert np.max(np.abs(data.y32 - data.y64)) <= data.y_tolerance
    assert np.max(np.abs(data.w32 - data.w64)) <= data.w_tolerance


def test_noisy_case_differs_from_clean_case():
    clean = load("tone_mixture")
    noisy = load("tone_mixture_snr40")

    np.testing.assert_array_equal(clean.x, noisy.x)   # same reference
    assert not np.array_equal(clean.e, noisy.e)       # noise reached e[n]
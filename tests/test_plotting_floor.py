import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from plotting import draw_measurement_floor, mean_finite_floor


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_floor_draws_line_and_band(ax):
    draw_measurement_floor(ax, 60.0, margin_db=3.0)
    assert [line.get_ydata()[0] for line in ax.get_lines()] == [60.0]
    assert len(ax.patches) == 1


def test_noise_free_draws_note_not_floor(ax):
    draw_measurement_floor(ax, None)
    assert len(ax.get_lines()) == 0
    assert len(ax.patches) == 0
    assert "No measurement floor" in ax.texts[0].get_text()


def test_infinite_floor_is_treated_as_noise_free(ax):
    draw_measurement_floor(ax, float("inf"))
    assert len(ax.get_lines()) == 0
    assert len(ax.texts) == 1


def test_data_limits_are_not_clipped(ax):
    """Rule 1: drawing the floor must not cut off data above it."""
    ax.plot([0, 1], [10.0, 90.0])
    draw_measurement_floor(ax, 60.0)
    assert ax.get_ylim()[1] >= 90.0


def test_mean_finite_floor_ignores_inf_and_none():
    rows = [
        {"measurement_floor_db": 59.0},
        {"measurement_floor_db": 61.0},
        {"measurement_floor_db": float("inf")},
        {"measurement_floor_db": None},
    ]
    assert mean_finite_floor(rows) == pytest.approx(60.0)
    assert mean_finite_floor([{"measurement_floor_db": float("inf")}]) is None
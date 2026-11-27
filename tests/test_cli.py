"""Black-box tests for anc_cli: arguments, input parsing, exit codes.

Exit codes (see the header of cpp/tools/anc_cli.cpp):
0 ok, 1 runtime error, 2 usage error, 3 controller fault.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

BASE = ["--length", "2", "--mu", "0.1", "--shat", "1.0"]


def run_cli(cli: Path, tmp_path: Path, rows: str | None, *options: str,
            base: list[str] | None = BASE):
    """Write `rows` to an input file (if given) and run the CLI on it."""
    io_path = tmp_path / "io.csv"
    if rows is not None:
        io_path.write_text(rows, encoding="utf-8")
    args = [str(cli),
            "--input", str(io_path),
            "--output-y", str(tmp_path / "y.csv"),
            "--output-weights", str(tmp_path / "w.csv"),
            *(base or []), *options]
    return subprocess.run(args, capture_output=True, text=True)


def read_y(tmp_path: Path) -> np.ndarray:
    return np.loadtxt(tmp_path / "y.csv", ndmin=1)


# --- normal operation --------------------------------------------------------

def test_help_prints_usage(cli):
    result = subprocess.run([str(cli), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage: anc_cli" in result.stdout


def test_hand_computed_run(cli, tmp_path):
    # Same numbers as test_streaming_update_follows_filtered_push.
    result = run_cli(cli, tmp_path, "1.0,0.5\n2.0,1.0\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fault=none fault_sample=0 samples=2"
    np.testing.assert_allclose(read_y(tmp_path), [-0.05, -0.6], atol=1e-7)
    np.testing.assert_allclose(np.loadtxt(tmp_path / "w.csv"), [-0.25, -0.10], atol=1e-7)


def test_header_line_is_skipped(cli, tmp_path):
    result = run_cli(cli, tmp_path, "x,e\n1.0,0.5\n2.0,1.0\n")
    assert result.returncode == 0, result.stderr
    assert read_y(tmp_path).shape == (2,)


def test_crlf_and_spaces_are_accepted(cli, tmp_path):
    result = run_cli(cli, tmp_path, "x,e\r\n 1.0 , 0.5 \r\n2.0,1.0\r\n")
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(read_y(tmp_path), [-0.05, -0.6], atol=1e-7)


# --- usage errors (exit 2) ---------------------------------------------------

@pytest.mark.parametrize("key, value, message", [
    ("--length", "12abc", "--length"),
    ("--length", "-1", "--length"),
    ("--length", "0", "filter_length"),
    ("--mu", "fast", "--mu"),
    ("--mu", "0.1abc", "--mu"),       # std::stod alone would accept this as 0.1
    ("--mu", "0", "step_size"),
    ("--shat", "1.0,x", "--shat"),
])
def test_invalid_option_values(cli, tmp_path, key, value, message):
    base = list(BASE)
    base[base.index(key) + 1] = value
    result = run_cli(cli, tmp_path, "1,1\n", base=base)
    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize("option, value", [
    ("--max-abs-error", "0"),
    ("--max-weight-norm", "-5"),
    ("--max-weight-norm", "nan"),
])
def test_invalid_limits(cli, tmp_path, option, value):
    result = run_cli(cli, tmp_path, "1,1\n", option, value)
    assert result.returncode == 2
    assert "limits" in result.stderr


def test_unknown_option(cli, tmp_path):
    result = run_cli(cli, tmp_path, "1,1\n", "--speed", "3")
    assert result.returncode == 2
    assert "unknown option: --speed" in result.stderr


def test_missing_required_option(cli, tmp_path):
    result = run_cli(cli, tmp_path, "1,1\n", base=["--length", "2", "--mu", "0.1"])
    assert result.returncode == 2
    assert "missing option --shat" in result.stderr


def test_option_without_value(cli, tmp_path):
    result = run_cli(cli, tmp_path, "1,1\n", "--delay")
    assert result.returncode == 2
    assert "missing value for --delay" in result.stderr


def test_usage_errors_are_reported_before_reading_input(cli, tmp_path):
    # The input file does not exist; the bad option must still win.
    result = run_cli(cli, tmp_path, None, base=["--length", "x", "--mu", "0.1", "--shat", "1"])
    assert result.returncode == 2


# --- runtime errors (exit 1) -------------------------------------------------

def test_missing_input_file(cli, tmp_path):
    result = run_cli(cli, tmp_path, None)
    assert result.returncode == 1
    assert "cannot open input file" in result.stderr


@pytest.mark.parametrize("rows, line", [
    ("1.0,0.5\n2.0\n", "line 2"),             # one value
    ("1.0,0.5\n2.0,1.0,3.0\n", "line 2"),     # three values
    ("1.0,0.5\n2.0,abc\n", "line 2"),         # not a number
    ("1.0,0.5\n2.0,1.0x\n", "line 2"),        # trailing garbage
    ("x,0.5\n", "line 1"),                    # half a header is not a header
])
def test_malformed_input_lines(cli, tmp_path, rows, line):
    result = run_cli(cli, tmp_path, rows)
    assert result.returncode == 1
    assert line in result.stderr


# --- controller faults (exit 3) ----------------------------------------------

def test_nan_input_latches_and_outputs_silence(cli, tmp_path):
    result = run_cli(cli, tmp_path, "1.0,0.5\nnan,1.0\n1.0,0.5\n")
    assert result.returncode == 3
    assert result.stdout.strip() == "fault=non_finite_input fault_sample=1 samples=3"
    np.testing.assert_allclose(read_y(tmp_path), [-0.05, 0.0, 0.0], atol=1e-7)


def test_error_limit_from_the_command_line(cli, tmp_path):
    result = run_cli(cli, tmp_path, "1.0,0.5\n1.0,20.0\n", "--max-abs-error", "10")
    assert result.returncode == 3
    assert "fault=error_limit fault_sample=1" in result.stdout


def test_weight_norm_limit_from_the_command_line(cli, tmp_path):
    # Twin of the C++ unit test: w = 4, 8, then 12 is rejected.
    rows = "1,-4\n1,-4\n1,-4\n"
    result = run_cli(cli, tmp_path, rows, "--max-weight-norm", "10",
                     base=["--length", "1", "--mu", "1", "--shat", "1"])
    assert result.returncode == 3
    assert "fault=weight_norm_limit fault_sample=2" in result.stdout
    np.testing.assert_allclose(np.loadtxt(tmp_path / "w.csv", ndmin=1), [8.0])
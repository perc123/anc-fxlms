"""Shared fixtures."""

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cli(tmp_path_factory) -> Path:
    """Build anc_cli once per test session in a temporary directory."""
    if shutil.which("cmake") is None:
        pytest.skip("cmake is not installed")

    build_dir = tmp_path_factory.mktemp("cpp_build")
    try:
        subprocess.run(
            ["cmake", "-S", str(PROJECT_ROOT / "cpp"), "-B", str(build_dir),
             "-DCMAKE_BUILD_TYPE=Release"],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "anc_cli"],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(f"C++ build failed:\n{error.stdout}\n{error.stderr}")

    return build_dir / "anc_cli"
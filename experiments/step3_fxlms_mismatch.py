from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from anc.controllers import FxLMSController
from anc.fxlms import run_fxlms
from anc.metrics import attenuation_db, rms
from anc.plant import ANCPlant
from anc.signals import generate_tone_mixture
from experiments.plotting import plot_fxlms_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]
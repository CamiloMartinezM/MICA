"""Configuration module for setting up paths."""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
print(f"ROOT_DIR path is: {ROOT_DIR}")

# TZINFO
TZINFO = ZoneInfo("Europe/Berlin")  # Set your timezone here

# Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set torch device
DEVICE = torch.device("cuda" if torch.cuda.device_count() >= 1 else "cpu")
print(f"Torch device: {DEVICE}, Total CUDA devices: {torch.cuda.device_count()}")

# Numpy Random Number Generator
RNG = np.random.default_rng()

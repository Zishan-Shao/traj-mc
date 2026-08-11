"""Project paths with optional environment-variable overrides."""

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent

DATA_DIR = Path(
    os.environ.get("QUANT_DLLM_DATA_DIR", WORKSPACE_DIR / "data")
).expanduser()
OUTPUT_DIR = Path(
    os.environ.get("QUANT_DLLM_OUTPUT_DIR", PROJECT_DIR / "output")
).expanduser()
LOG_DIR = Path(
    os.environ.get("QUANT_DLLM_LOG_DIR", PROJECT_DIR / "log")
).expanduser()

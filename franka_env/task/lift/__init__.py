"""Lift task configuration and components."""

import sys
from pathlib import Path

# Add parent directories to path for imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # franka_env root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import lift task configuration
from .env_cfg import FrankaShadowLiftEnvCfg

__all__ = ['FrankaShadowLiftEnvCfg']

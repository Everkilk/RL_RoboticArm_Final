"""Franka environment package for robotic manipulation tasks."""

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # franka_env root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# Import shared manager
from manager import ManagerRLGoalEnv

# Import task-specific configurations
from task.lift import FrankaShadowLiftEnvCfg
from task.lift_orientation import FrankaShadowLiftOrientationEnvCfg

# Export public API
__all__ = ['ManagerRLGoalEnv', 'FrankaShadowLiftEnvCfg', 'FrankaShadowLiftOrientationEnvCfg']
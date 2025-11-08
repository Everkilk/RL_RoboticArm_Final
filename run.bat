@echo off
REM Training script launcher for Franka Robotic Arm with Isaac Sim
REM This script ensures CUDA libraries are available and uses Isaac Sim's Python

REM CRITICAL: Deactivate ALL conda environments to force Isaac Sim Python usage
:deactivate_conda
if defined CONDA_DEFAULT_ENV (
    echo [INFO] Deactivating conda environment: %CONDA_DEFAULT_ENV%
    call conda deactivate
    goto deactivate_conda
)
if defined CONDA_PREFIX (
    echo [INFO] Clearing conda environment variables...
    set CONDA_PREFIX=
    set CONDA_DEFAULT_ENV=
    set CONDA_PYTHON_EXE=
    set CONDA_PROMPT_MODIFIER=
)

echo ========================================
echo Franka Robotic Arm Training Launcher
echo ========================================
echo.

REM Optimize CPU threading for multi-environment training
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set NUMEXPR_NUM_THREADS=4
echo [INFO] CPU threading optimized (OMP/MKL threads = 4)

REM Cleanup old experiments automatically (keep last 3)
echo [INFO] Cleaning up old experiment folders...
for /f "skip=3 delims=" %%D in ('dir /AD /B /O-D runs\exp* 2^>nul') do (
    echo [INFO] Removing old experiment: %%D
    rd /s /q "runs\%%D" 2>nul
)
echo.

REM Check if IsaacLab is available
if not exist "C:\IsaacLab\isaaclab.bat" (
    echo [ERROR] IsaacLab not found at C:\IsaacLab
    echo Please ensure Isaac Sim and IsaacLab are properly installed.
    pause
    exit /b 1
)

REM Run the training script using Isaac Sim's Python
echo [INFO] Starting training with Isaac Sim environment...
echo [INFO] Python from: C:\IsaacLab\_isaac_sim\python.bat
echo [INFO] Number of environments: 40 (adaptive scaling enabled)
echo [INFO] Random seed: 42 (for reproducibility)
echo [INFO] Mode: Headless (no GUI for faster training)
echo.

C:\IsaacLab\isaaclab.bat -p franka_train.py --num-envs 40 --seed 42 --headless

echo.
echo ========================================
echo Training Complete
echo ========================================
pause

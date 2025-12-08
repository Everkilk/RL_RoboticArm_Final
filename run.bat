REM Optimize CPU threading for multi-environment training
set OMP_NUM_THREADS=6
set MKL_NUM_THREADS=6
set NUMEXPR_NUM_THREADS=6

REM Cleanup old experiments automatically (keep last 3)
echo [INFO] Cleaning up old experiment folders...
for /f "skip=10 delims=" %%D in ('dir /AD /B /O-D runs\exp* 2^>nul') do (
    echo [INFO] Removing old experiment: %%D
    rd /s /q "runs\%%D" 2>nul
)
echo.

REM Check if IsaacLab is available
if not exist "C:\IsaacLab\isaaclab.bat" (
    echo [ERROR] IsaacLab not found at C:\Users\PC\IsaacLab
    echo Please ensure Isaac Sim and IsaacLab are properly installed.
    pause
    exit /b 1
)

REM Run the training script using Isaac Sim's Python
C:\IsaacLab\isaaclab.bat -p franka_rher_sac_train.py --num-envs 8 --seed 42 --headless
C:\IsaacLab\isaaclab.bat -p franka_rher_ddpg_train.py --num-envs 8 --seed 42 --headless --resume_path "C:\isaac-projects\Test\runs\exp1"
C:\IsaacLab\isaaclab.bat -p franka_her_ddpg_train.py --num-envs 8 --seed 42 --headless --resume_path "C:\isaac-projects\Test\runs\exp3"
C:\IsaacLab\isaaclab.bat -p franka_her_sac_train.py --num-envs 8 --seed 42 --headless --resume_path "C:\isaac-projects\Test\runs\exp2"

echo.
echo ========================================
echo Training Complete
echo ========================================
pause

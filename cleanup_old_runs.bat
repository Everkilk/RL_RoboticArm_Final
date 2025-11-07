@echo off
REM Cleanup old experiment folders to free up disk space

echo ========================================
echo Experiment Folder Cleanup
echo ========================================
echo.

cd /d "%~dp0"

if not exist "runs" (
    echo [INFO] No 'runs' folder found. Nothing to clean.
    pause
    exit /b 0
)

echo Current experiments in runs folder:
dir /AD /B runs
echo.

set /p CONFIRM="Delete ALL old experiment folders? This will keep only exp0. (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo Cleanup cancelled.
    pause
    exit /b 0
)

echo.
echo Deleting old experiments...

REM Keep only exp0, delete everything else
for /d %%D in (runs\exp*) do (
    if not "%%~nxD"=="exp0" (
        echo Removing: %%~nxD
        rd /s /q "%%D"
    )
)

echo.
echo ========================================
echo Cleanup Complete!
echo ========================================
echo Remaining experiments:
dir /AD /B runs
echo.
pause

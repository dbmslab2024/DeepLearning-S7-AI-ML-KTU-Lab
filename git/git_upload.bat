@echo off
rem Batch script for git upload
rem Usage: git_upload.bat

echo Current directory: %CD%

rem Change to the directory where this batch file is located
cd /d "%~dp0"

echo Changed to script directory: %CD%

rem Check if we're in a git repository
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo Error: Not in a git repository!
    pause
    exit /b 1
)

echo Adding files to git...
git add .

echo Committing changes...
git commit -m "Updated files via batch script"

echo Pushing to origin main...
git push origin main

if errorlevel 1 (
    echo Git upload failed!
    pause
    exit /b 1
) else (
    echo Git upload completed successfully!
)

pause

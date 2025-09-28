# PowerShell script for git upload
# Usage: .\git_upload.ps1

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Change to the script directory (which should be the git repo root)
Set-Location $ScriptDir

# Check if we're in a git repository
try {
    git rev-parse --git-dir 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Not in a git repository"
    }
} catch {
    Write-Error "Error: Not in a git repository!"
    exit 1
}

Write-Host "Current directory: $(Get-Location)" -ForegroundColor Green
Write-Host "Adding files to git..." -ForegroundColor Yellow
git add .

Write-Host "Committing changes..." -ForegroundColor Yellow
git commit -m "Updated files via PowerShell script"

Write-Host "Pushing to origin main..." -ForegroundColor Yellow
git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "Git upload completed successfully!" -ForegroundColor Green
} else {
    Write-Error "Git upload failed!"
    exit 1
}

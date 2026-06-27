$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "Setting up Claim Denial Prediction project..." -ForegroundColor Cyan

if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

$pythonPath = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    throw "Virtual environment Python was not found at $pythonPath"
}

Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $pythonPath -m pip install --upgrade pip

Write-Host "Installing requirements..." -ForegroundColor Yellow
& $pythonPath -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run .\run_pipeline.ps1 to rebuild outputs, or .\run_app.ps1 to start the demo app." -ForegroundColor Cyan

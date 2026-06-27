$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "Starting FastAPI on http://localhost:8000 ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$PSScriptRoot'; python -m uvicorn app.step12_fastapi:app --reload --port 8000"
)

Start-Sleep -Seconds 4

Write-Host "Starting Streamlit dashboard..." -ForegroundColor Cyan
python -m streamlit run app\step13_Dashboard.py

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "Running Project 01 data + ML + NLP pipeline..." -ForegroundColor Cyan

$steps = @(
    "step01_eda.py",
    "step02_null_detection.py",
    "step03_duplicate_outlier.py",
    "step04_cleaning.py",
    "step05_credentials.py",
    "step06_encoding.py",
    "step07_premodel_eda.py",
    "step08_model.py",
    "step09_rarc_taxonomy.py"
)

foreach ($step in $steps) {
    Write-Host "`n=== Running $step ===" -ForegroundColor Yellow
    python $step
}

if ((Test-Path "real_rarc_codes.csv") -and -not (Test-Path "nlp_outputs\real_rarc_codes.csv")) {
    New-Item -ItemType Directory -Force -Path "nlp_outputs" | Out-Null
    Copy-Item "real_rarc_codes.csv" "nlp_outputs\real_rarc_codes.csv"
    Write-Host "`nCopied real_rarc_codes.csv to nlp_outputs\real_rarc_codes.csv" -ForegroundColor Green
}

$nlpSteps = @(
    "step10_synthetic_rarc_data.py",
    "step11_nlp_classifier.py"
)

foreach ($step in $nlpSteps) {
    Write-Host "`n=== Running $step ===" -ForegroundColor Yellow
    python $step
}

Write-Host "`nPipeline complete." -ForegroundColor Green
Write-Host "Next: run .\run_app.ps1 to start API + dashboard." -ForegroundColor Cyan

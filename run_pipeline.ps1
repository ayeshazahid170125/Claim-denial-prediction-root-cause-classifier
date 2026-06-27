$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "Running Project 01 data + ML + NLP pipeline..." -ForegroundColor Cyan

$steps = @(
    "src\step01_eda.py",
    "src\step02_null_detection.py",
    "src\step03_duplicate_outlier.py",
    "src\step04_cleaning.py",
    "src\step05_credentials.py",
    "src\step06_encoding.py",
    "src\step07_premodel_eda.py",
    "src\step08_model.py",
    "src\step09_rarc_taxonomy.py"
)

foreach ($step in $steps) {
    Write-Host "`n=== Running $step ===" -ForegroundColor Yellow
    python $step
}

if ((Test-Path "real_rarc_codes.csv") -and -not (Test-Path "outputs\nlp\real_rarc_codes.csv")) {
    New-Item -ItemType Directory -Force -Path "outputs\nlp" | Out-Null
    Copy-Item "real_rarc_codes.csv" "outputs\nlp\real_rarc_codes.csv"
    Write-Host "`nCopied real_rarc_codes.csv to outputs\nlp\real_rarc_codes.csv" -ForegroundColor Green
}

$nlpSteps = @(
    "src\step10_synthetic_rarc_data.py",
    "src\step11_nlp_classifier.py"
)

foreach ($step in $nlpSteps) {
    Write-Host "`n=== Running $step ===" -ForegroundColor Yellow
    python $step
}

Write-Host "`nPipeline complete." -ForegroundColor Green
Write-Host "Next: run .\run_app.ps1 to start API + dashboard." -ForegroundColor Cyan

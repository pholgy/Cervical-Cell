param(
    [switch]$Install
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot

try {
    function Invoke-Step {
        param(
            [Parameter(Mandatory = $true)][string]$Name,
            [Parameter(Mandatory = $true)][scriptblock]$Command
        )

        Write-Host ""
        Write-Host "==> $Name"
        & $Command
    }

    Invoke-Step "Python syntax check" {
        python -m py_compile `
            config.py `
            api/index.py `
            src/data_loader.py `
            src/train_model.py `
            src/api/main.py `
            src/api/main_gemini.py `
            quick_train.py `
            train_100_epochs.py `
            train_improved.py `
            train_optimized.py `
            train_resnet50.py `
            test_simple.py `
            test_data_loading.py `
            verify_structure.py
    }

    if ($Install) {
        Invoke-Step "Install backend test dependencies" {
            python -m pip install -r requirements-dev.txt
        }
    }

    try {
        python -c "import pytest, httpx" | Out-Null
    }
    catch {
        throw "Backend test dependencies are missing. Run 'python -m pip install -r requirements-dev.txt' or '.\scripts\check.ps1 -Install'."
    }

    Invoke-Step "Backend tests" {
        python -m pytest tests -q
    }

    Invoke-Step "Lightweight structure check" {
        python verify_structure.py
    }

    if ($Install -and -not (Test-Path "frontend/node_modules")) {
        Invoke-Step "Install frontend dependencies" {
            npm --prefix frontend ci
        }
    }

    if (-not (Test-Path "frontend/node_modules")) {
        throw "frontend/node_modules is missing. Run 'npm --prefix frontend ci' or '.\scripts\check.ps1 -Install'."
    }

    Invoke-Step "Frontend production build" {
        npm --prefix frontend run build
    }
}
finally {
    Pop-Location
}

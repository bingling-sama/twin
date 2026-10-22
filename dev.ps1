<#
.SYNOPSIS
    Starts Twin backend (Uvicorn) and frontend (Vite) concurrently on Windows PowerShell.
.PARAMETER BackendOnly
    Start only the backend service.
.PARAMETER FrontendOnly
    Start only the frontend service.
.PARAMETER Port
    Backend port (default: 8000).
.PARAMETER FrontPort
    Frontend port (default: 5173).
#>
[CmdletBinding()]
param(
    [switch]$BackendOnly,
    [switch]$FrontendOnly,
    [int]$Port = 8000,
    [int]$FrontPort = 5173
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# Environment variables to avoid OpenMP duplicate runtime crashes
$env:OMP_NUM_THREADS = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Locate Python
$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        $PythonExe = $cmd.Source
    }
}

if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found in .venv or PATH."
    exit 1
}

Write-Host "Using Python: $PythonExe" -ForegroundColor Cyan

$Processes = @()

function Stop-AllProcesses {
    Write-Host "`nStopping Twin services..." -ForegroundColor Yellow
    foreach ($p in $Processes) {
        if ($p -and -not $p.HasExited) {
            try {
                Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
                Write-Host "Terminated process $($p.Id)" -ForegroundColor DarkGray
            } catch {}
        }
    }
}

try {
    if (-not $FrontendOnly) {
        Write-Host "Starting Twin Backend on http://localhost:$Port ..." -ForegroundColor Green
        $backendProcess = Start-Process -FilePath $PythonExe `
            -ArgumentList "-m uvicorn twin.main:app --reload --reload-dir src/twin --host 0.0.0.0 --port $Port" `
            -WorkingDirectory $ScriptDir `
            -PassThru
        $Processes += $backendProcess
    }

    if (-not $BackendOnly) {
        Write-Host "Starting Twin Frontend on http://localhost:$FrontPort ..." -ForegroundColor Green
        $webDir = Join-Path $ScriptDir "src\example\web"
        $pnpmExe = (Get-Command pnpm.cmd, pnpm -ErrorAction SilentlyContinue | Select-Object -First 1).Source
        if (-not $pnpmExe) {
            Write-Error "pnpm not found in PATH."
            exit 1
        }
        $frontendProcess = Start-Process -FilePath $pnpmExe `
            -ArgumentList "dev --host" `
            -WorkingDirectory $webDir `
            -PassThru
        if ($frontendProcess) {
            $Processes += $frontendProcess
        }
    }

    Write-Host "`nBoth services running. Press Ctrl+C to terminate all services.`n" -ForegroundColor Cyan

    while ($true) {
        Start-Sleep -Seconds 1
        foreach ($p in $Processes) {
            if ($p.HasExited) {
                Write-Warning "Process $($p.Id) exited with code $($p.ExitCode)."
                exit $p.ExitCode
            }
        }
    }
}
finally {
    Stop-AllProcesses
}

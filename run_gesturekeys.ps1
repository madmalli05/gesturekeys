param(
    [ValidateSet("desktop", "api", "tk")]
    [string]$Mode = "desktop",
    [int]$Camera = -1,
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8000,
    [switch]$DisableKeys,
    [switch]$NoFallback,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Get-PythonCommand {
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return ,$venvPython
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3")
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }

    throw "Python 3.10+ was not found on PATH."
}

$pythonCommand = @(Get-PythonCommand)
$pythonExe = $pythonCommand[0]
$pythonArgs = @()

if ($pythonCommand.Count -gt 1) {
    $pythonArgs = $pythonCommand[1..($pythonCommand.Count - 1)]
}

if (-not $SkipInstall) {
    & $pythonExe @pythonArgs -m pip install -r requirements.txt
}

$appArgs = @("gesture_keyboard.py", "--mode", $Mode, "--host", $ApiHost, "--port", $ApiPort)

if ($Camera -ge 0) {
    $appArgs += @("--camera", $Camera)
}

if ($DisableKeys) {
    $appArgs += "--disable-keys"
}

if ($NoFallback) {
    $appArgs += "--no-fallback"
}

& $pythonExe @pythonArgs @appArgs

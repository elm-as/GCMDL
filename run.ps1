# PowerShell 1-Click Runner for GCMDL
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

$PythonExe = "C:\Users\elmas\AppData\Local\Programs\Python\Python310\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "GCMDL 2026: Global Crypto Market Dynamics & Liquidity" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

& $PythonExe main.py @args


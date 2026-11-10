@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo GCMDL 2026: Global Crypto Market Dynamics & Liquidity
echo ========================================================
"C:\Users\elmas\AppData\Local\Programs\Python\Python310\python.exe" main.py %*
pause

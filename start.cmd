@echo off
rem Starts the predict.fun 5-minute predictor and restarts it if it stops.  Open http://localhost:8790
cd /d "%~dp0"
:loop
"C:\Program Files\nodejs\node.exe" server.js
echo %date% %time% predictor stopped, restarting in 5s
timeout /t 5 /nobreak >nul
goto loop

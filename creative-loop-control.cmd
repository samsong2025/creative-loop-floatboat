@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Restricted launcher for Floatboat. It accepts only: status | start | rebuild.
rem It never accepts an arbitrary command, path, compose file, or Docker argument.
set "ROOT=%~dp0"
set "COMPOSE=%ROOT%compose.yaml"
set "ACTION=%~1"

if /I "%ACTION%"=="status" goto :status
if /I "%ACTION%"=="start" goto :start
if /I "%ACTION%"=="rebuild" goto :rebuild

echo Usage: creative-loop-control.cmd ^<status^|start^|rebuild^>
exit /b 64

:assert_files
if not exist "%COMPOSE%" (
  echo ERROR: Compose file is missing: %COMPOSE%
  exit /b 2
)
where docker >nul 2>nul
if errorlevel 1 (
  echo ERROR: Docker CLI is not available on PATH.
  exit /b 3
)
exit /b 0

:wait_for_docker
set /a WAITED=0
:wait_loop
docker info >nul 2>nul
if not errorlevel 1 exit /b 0
if !WAITED! GEQ 180 (
  echo ERROR: Docker Desktop did not become ready within 180 seconds.
  exit /b 4
)
timeout /t 3 /nobreak >nul
set /a WAITED+=3
goto :wait_loop

:start_desktop_if_needed
docker info >nul 2>nul
if not errorlevel 1 exit /b 0
if exist "D:\Docker\DockerDesktop\Docker Desktop.exe" (
  start "Creative Loop Docker Desktop" "D:\Docker\DockerDesktop\Docker Desktop.exe"
) else if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" (
  start "Creative Loop Docker Desktop" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
) else (
  echo ERROR: Docker Desktop is not running and its executable was not found.
  exit /b 5
)
call :wait_for_docker
exit /b %errorlevel%

:status
call :assert_files
if errorlevel 1 exit /b %errorlevel%
docker info >nul 2>nul
if errorlevel 1 (
  echo STATUS: Docker Desktop is not running.
  exit /b 10
)
docker compose -f "%COMPOSE%" ps
exit /b %errorlevel%

:start
call :assert_files
if errorlevel 1 exit /b %errorlevel%
call :start_desktop_if_needed
if errorlevel 1 exit /b %errorlevel%
docker compose -f "%COMPOSE%" up -d
if errorlevel 1 exit /b %errorlevel%
docker compose -f "%COMPOSE%" ps
exit /b %errorlevel%

:rebuild
call :assert_files
if errorlevel 1 exit /b %errorlevel%
call :start_desktop_if_needed
if errorlevel 1 exit /b %errorlevel%
docker compose -f "%COMPOSE%" up -d --build creative-loop-api
if errorlevel 1 exit /b %errorlevel%
docker compose -f "%COMPOSE%" ps
exit /b %errorlevel%
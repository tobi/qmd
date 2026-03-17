@ECHO off
SETLOCAL
CALL :find_dp0

SET _prog=
IF EXIST "%dp0%\node.exe" (
  SET "_prog=%dp0%\node.exe"
) ELSE (
  SET _prog=node
  SET PATHEXT=%PATHEXT:;.JS;=;%
)

"%_prog%" "%dp0%\qmd.js" %*
ENDLOCAL
EXIT /b %errorlevel%
:find_dp0
SET dp0=%~dp0
EXIT /b

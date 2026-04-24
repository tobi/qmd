@echo off
:: Windows CMD launcher for QMD.
::
:: npm auto-generates a qmd.cmd wrapper that calls /bin/sh to run bin/qmd.
:: When MCP clients (e.g. Claude Code) spawn qmd as a stdio subprocess via
:: Node.js child_process.spawn(), /bin/sh is not reliably available — Git
:: for Windows is often absent from the subprocess PATH even when present
:: in interactive shells. This causes the MCP server to fail silently.
::
:: This hand-crafted launcher mirrors the runtime-detection logic in bin/qmd
:: (bun.lock → bun, package-lock.json → node, fallback → node) without
:: requiring a POSIX shell, making stdio MCP spawning work on any Windows
:: machine regardless of Git installation.
::
:: npm places this file in the global bin directory. The package root is
:: always at node_modules\@tobilu\qmd relative to that directory (matching
:: the path npm itself uses in auto-generated wrappers).

setlocal enabledelayedexpansion

set "PKG_DIR=%~dp0node_modules\@tobilu\qmd"

:: For local dev installs (npm link, direct repo use), fall back to
:: the parent of the bin/ directory.
if not exist "%PKG_DIR%\package.json" (
    set "PKG_DIR=%~dp0.."
)

if exist "%PKG_DIR%\package-lock.json" (
    node "%PKG_DIR%\dist\cli\qmd.js" %*
) else if exist "%PKG_DIR%\bun.lock" (
    bun "%PKG_DIR%\dist\cli\qmd.js" %*
) else if exist "%PKG_DIR%\bun.lockb" (
    bun "%PKG_DIR%\dist\cli\qmd.js" %*
) else (
    node "%PKG_DIR%\dist\cli\qmd.js" %*
)

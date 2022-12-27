@echo off

SET FORMATTER_DIR=%~dp0

%JULIA_184% --project=%FORMATTER_DIR% %FORMATTER_DIR%\format.jl
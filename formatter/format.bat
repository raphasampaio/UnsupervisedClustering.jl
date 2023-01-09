@echo off

SET FORMATTER_PATH=%~dp0

%JULIA_184% --project=%FORMATTER_PATH% %FORMATTER_PATH%\format.jl
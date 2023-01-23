@echo off

SET FORMATTER_PATH=%~dp0
DEL "%BASEPATH%\Manifest.toml"

%JULIA_185% --color=yes --project=%FORMATTER_PATH% %FORMATTER_PATH%\format.jl
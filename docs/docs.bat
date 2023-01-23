@echo off

SET DOCUMENTER_PATH=%~dp0
DEL "%DOCUMENTER_PATH%\Manifest.toml"

%JULIA_185% --color=yes --project=%DOCUMENTER_PATH% %DOCUMENTER_PATH%\make.jl
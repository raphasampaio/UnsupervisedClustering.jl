@echo off

SET REVISE_PATH=%~dp0
DEL "%REVISE_PATH%\Manifest.toml"

%JULIA_184% --color=yes --project=%REVISE_PATH% --load=%REVISE_PATH%\revise.jl

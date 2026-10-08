@echo off

SET BASE_PATH=%~dp0

CALL julia --project=%BASE_PATH% --interactive --load=%BASE_PATH%\revise.jl

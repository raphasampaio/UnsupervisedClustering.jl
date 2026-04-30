@echo off

SET BASE_PATH=%~dp0

CALL julia +1.12 --project=%BASE_PATH% --interactive --load=%BASE_PATH%\revise.jl

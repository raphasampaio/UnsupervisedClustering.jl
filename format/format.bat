@echo off

SET BASE_PATH=%~dp0

CALL julia --project=%BASE_PATH% %BASE_PATH%\format.jl
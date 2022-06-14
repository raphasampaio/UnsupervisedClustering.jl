@echo off

SET FORMATTER_DIR=%~dp0

julia --project=%FORMATTER_DIR% %FORMATTER_DIR%\format.jl

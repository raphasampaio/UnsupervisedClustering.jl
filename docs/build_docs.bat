@echo off

SET DOCUMENTER_DIR=%~dp0

julia --project=%DOCUMENTER_DIR% %DOCUMENTER_DIR%\make.jl

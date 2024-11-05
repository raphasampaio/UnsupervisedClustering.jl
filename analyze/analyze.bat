@echo off

SET BASEPATH=%~dp0

CALL "%JULIA_1105%" --project=%BASEPATH% --load=%BASEPATH%\analyze.jl

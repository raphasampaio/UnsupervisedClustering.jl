@echo off

SET BASEPATH=%~dp0

CALL "%JULIA_1104%" --project=%BASEPATH% %BASEPATH%\format.jl
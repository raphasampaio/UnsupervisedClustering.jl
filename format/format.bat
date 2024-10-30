@echo off

SET BASEPATH=%~dp0

CALL "%JULIA_1111%" --project=%BASEPATH% %BASEPATH%\format.jl
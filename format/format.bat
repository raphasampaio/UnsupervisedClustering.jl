@echo off

SET BASEPATH=%~dp0

CALL "%JULIA_1113%" --project=%BASEPATH% %BASEPATH%\format.jl
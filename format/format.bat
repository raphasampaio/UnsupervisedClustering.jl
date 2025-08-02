@echo off

SET BASEPATH=%~dp0

CALL julia +1.11 --project=%BASEPATH% %BASEPATH%\format.jl
@echo off

SET BASEPATH=%~dp0

CALL "%JULIA_194%" --project=%BASEPATH% --interactive --load=%BASEPATH%\revise.jl

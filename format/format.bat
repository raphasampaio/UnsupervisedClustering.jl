@echo off

SET BASEPATH=%~dp0

"%JULIA_193%" --project=%BASEPATH% %BASEPATH%\format.jl
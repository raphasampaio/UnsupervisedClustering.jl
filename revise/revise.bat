@echo off

SET BASEPATH=%~dp0

"%JULIA_193%" --project=%BASEPATH% --interactive --load=%BASEPATH%\revise.jl

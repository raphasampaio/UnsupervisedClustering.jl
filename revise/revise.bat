@echo off

SET BASEPATH=%~dp0

%JULIA_192% --project=%BASEPATH% --interactive --load=%BASEPATH%\revise.jl

@echo off

SET BASEPATH=%~dp0

%JULIA_192% --project=%BASEPATH% --load=%BASEPATH%\revise.jl

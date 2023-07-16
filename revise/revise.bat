@echo off

SET BASEPATH=%~dp0

%JULIA_192% --color=yes --project=%BASEPATH% --load=%BASEPATH%\revise.jl

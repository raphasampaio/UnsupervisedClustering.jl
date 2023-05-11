@echo off

SET BASEPATH=%~dp0

%JULIA_190% --color=yes --project=%BASEPATH% --load=%BASEPATH%\revise.jl

@echo off

SET BASEPATH=%~dp0

%JULIA_185% --color=yes --project=%BASEPATH% --load=%BASEPATH%\revise.jl

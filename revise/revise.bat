@echo off

SET BASEPATH=%~dp0
DEL "%BASEPATH%\Manifest.toml"

%JULIA_185% --color=yes --project=%BASEPATH% --load=%BASEPATH%\revise.jl
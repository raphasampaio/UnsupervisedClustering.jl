@echo off

SET BASEPATH=%~dp0
DEL "%BASEPATH%\..\Manifest.toml"

%JULIA_185% --project=%BASEPATH%\.. -e "import Pkg; Pkg.test()"
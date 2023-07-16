@echo off

SET BASEPATH=%~dp0
DEL "%BASEPATH%\..\Manifest.toml"

%JULIA_192% --project=%BASEPATH%\.. -e "import Pkg; Pkg.test()"
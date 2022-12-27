@echo off

SET BASEPATH=%~dp0

%JULIA_184% --project=%BASEPATH%\.. -e "import Pkg; Pkg.test()"
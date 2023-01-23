@echo off

SET BASEPATH=%~dp0

%JULIA_185% --project=%BASEPATH%\.. -e "import Pkg; Pkg.test()"
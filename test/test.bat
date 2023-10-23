@echo off

SET BASEPATH=%~dp0

CALL "%JULIA_193%" --project=%BASEPATH%\.. -e "import Pkg; Pkg.test()"
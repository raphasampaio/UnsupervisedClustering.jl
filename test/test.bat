@echo off

SET BASEPATH=%~dp0

CALL "%JULIA_194%" --project=%BASEPATH%\.. -e "import Pkg; Pkg.test()"
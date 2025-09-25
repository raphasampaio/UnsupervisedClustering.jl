@echo off

SET BASEPATH=%~dp0

IF "%~1"=="" (
    CALL julia +1.11 --project=%BASEPATH%\.. -e "import Pkg; Pkg.test()"
) ELSE (
    CALL julia +1.11 --project=%BASEPATH%\.. %BASEPATH%\runtests.jl %1
)
@echo off

SET BASE_PATH=%~dp0

IF "%~1"=="" (
    CALL julia +1.12 --project=%BASE_PATH%\.. -e "import Pkg; Pkg.test()"
) ELSE (
    CALL julia +1.12 --project=%BASE_PATH%\.. %BASE_PATH%\runtests.jl %1
)
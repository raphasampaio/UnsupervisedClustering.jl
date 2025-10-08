@echo off

SET BASEPATH=%~dp0

CALL julia +1.12 --project=%BASEPATH% -e "using Pkg; Pkg.develop(PackageSpec(path=dirname(pwd()))); Pkg.instantiate()"
CALL julia +1.12 --project=%BASEPATH% %BASEPATH%\make.jl
@echo off

SET BASEPATH=%~dp0

CALL julia +1.11 --project -e "using Pkg; Pkg.develop(PackageSpec(path=dirname(pwd()))); Pkg.instantiate()"
CALL julia +1.11 --project %BASEPATH%\make.jl
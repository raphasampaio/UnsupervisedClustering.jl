@echo off

SET BASE_PATH=%~dp0

CALL julia +1.12 --project=%BASE_PATH% -e "using Pkg; Pkg.develop(PackageSpec(path=dirname(pwd()))); Pkg.instantiate()"
CALL julia +1.12 --project=%BASE_PATH% %BASE_PATH%\make.jl
@echo off

SET BASEPATH=%~dp0
SET OMP_NUM_THREADS=1

%JULIA_184% --project=%BASEPATH%\.. -e "import Pkg; Pkg.test()"
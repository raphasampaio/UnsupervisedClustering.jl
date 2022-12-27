@echo off

@REM Colocar variaveis de ambiente se necessário
SET BASEPATH=%~dp0
SET REVISE_PATH="%BASEPATH%\revise"
SET OMP_NUM_THREADS=1

%JULIA_184% --color=yes --project=%REVISE_PATH% --load=%REVISE_PATH%\revise.jl

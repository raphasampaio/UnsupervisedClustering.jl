#!/bin/bash

$JULIA_1111 --project -e "using Pkg; Pkg.develop(PackageSpec(path=dirname(pwd()))); Pkg.instantiate()"
$JULIA_1111 --project make.jl
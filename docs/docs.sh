#!/bin/bash

$JULIA_1105 --project -e "using Pkg; Pkg.develop(PackageSpec(path=dirname(pwd()))); Pkg.instantiate()"
$JULIA_1105 --project make.jl
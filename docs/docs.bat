CALL "%JULIA_1111%" --project -e "using Pkg; Pkg.develop(PackageSpec(path=dirname(pwd()))); Pkg.instantiate()"
CALL "%JULIA_1111%" --project make.jl
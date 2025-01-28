CALL "%JULIA_1113%" --project -e "using Pkg; Pkg.develop(PackageSpec(path=dirname(pwd()))); Pkg.instantiate()"
CALL "%JULIA_1113%" --project make.jl
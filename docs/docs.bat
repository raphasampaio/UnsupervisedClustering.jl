"%JULIA_193%" --project -e "using Pkg; Pkg.develop(PackageSpec(path=dirname(pwd()))); Pkg.instantiate()"
"%JULIA_193%" --project make.jl
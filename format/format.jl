import Pkg
Pkg.instantiate()

using JuliaFormatter

if format(dirname(@__DIR__))
    exit(0)
else
    @error "Some files have not been formatted!"
    exit(1)
end

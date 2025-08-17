import Pkg
Pkg.instantiate()

using JuliaFormatter

if format(dirname(dirname(@__FILE__)))
    exit(0)
else
    @error "Some files have not been formatted!"
    exit(1)
end

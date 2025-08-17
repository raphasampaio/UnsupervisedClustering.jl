import Pkg
Pkg.instantiate()

using JuliaFormatter



if format(dirname(dirname(@__FILE__)))
    @info "All files have been formatted!"
    exit(0)
else
    @error "Some files have not been formatted!"
    exit(1)
end

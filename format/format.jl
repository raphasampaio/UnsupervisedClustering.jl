import Pkg
Pkg.instantiate()

using JuliaFormatter

formatted = format(dirname(@__FILE__))

if formatted
    @info "All files have been formatted!"
    exit(0)
else
    @error "Some files have not been formatted!"
    exit(1)
end

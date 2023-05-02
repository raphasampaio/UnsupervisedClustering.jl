import Pkg
Pkg.instantiate()

using JuliaFormatter

for folder in ["docs", "format", "profiling", "revise", "src", "test"]
    format(joinpath(dirname(@__DIR__), folder))
end

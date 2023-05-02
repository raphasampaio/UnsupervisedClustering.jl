import Pkg
Pkg.instantiate()

using JuliaFormatter

for folder in ["docs", "format", "profiling", "revise", "src", "test"]
    println("Formatted " * folder * ": " * format(joinpath(dirname(@__DIR__), folder)))
end

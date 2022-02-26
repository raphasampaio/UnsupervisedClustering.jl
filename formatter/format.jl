import Pkg
Pkg.instantiate()

using JuliaFormatter

formatted_src = format(joinpath(dirname(@__DIR__), "src"))
formatted_test = format(joinpath(dirname(@__DIR__), "test"))
println("src folder formatted: $formatted_src")
println("test folder formatted: $formatted_test")

import Pkg
Pkg.instantiate()

using JET

Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()

using UnsupervisedClustering

package_path = dirname(@__DIR__)
module_path = joinpath(package_path, "src/UnsupervisedClustering.jl")

@show report_file(module_path; analyze_from_definitions = true)

# @show report_package(UnsupervisedClustering)

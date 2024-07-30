import Pkg
Pkg.instantiate()

using JET

Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()

using UnsupervisedClustering

package_path = dirname(@__DIR__)
module_path = joinpath(package_path, "src/UnsupervisedClustering.jl")

report = report_file(module_path; analyze_from_definitions = true)
println("Report 1")
@show report

report = report_package(UnsupervisedClustering)
println("Report 2")
@show report

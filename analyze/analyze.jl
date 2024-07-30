import Pkg
Pkg.instantiate()

using JET

Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()

using UnsupervisedClustering

@show report_package(UnsupervisedClustering)

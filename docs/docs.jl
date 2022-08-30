import Pkg
Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()
using UnsupervisedClustering

Pkg.activate(@__DIR__)
Pkg.instantiate()
using Documenter

makedocs(;
    modules = [UnsupervisedClustering],
    doctest = true,
    clean = true,
    format = Documenter.HTML(mathengine = Documenter.MathJax2()),
    sitename = "UnsupervisedClustering.jl",
    pages = ["Home" => "index.md"],
)

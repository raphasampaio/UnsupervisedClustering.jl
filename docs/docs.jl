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
    format = Documenter.HTML(
        # assets = ["assets/favicon.ico"],
        mathengine = Documenter.MathJax2()
    ),
    sitename = "UnsupervisedClustering.jl",
    authors = "Raphael Araujo Sampaio, and contributors",
    pages = [
        "Home" => "index.md",
    ],
)

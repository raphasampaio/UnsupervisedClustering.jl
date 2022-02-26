import Pkg
Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()
using UnsupervisedLearning

Pkg.activate(@__DIR__)
Pkg.instantiate()
using Documenter

makedocs(;
    modules = [UnsupervisedLearning],
    doctest = true,
    clean = true,
    format = Documenter.HTML(mathengine = Documenter.MathJax2()),
    sitename = "UnsupervisedLearning.jl",
    pages = ["Home" => "index.md"],
)

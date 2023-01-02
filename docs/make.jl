import Pkg
Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()
using UnsupervisedClustering

using Documenter

DocMeta.setdocmeta!(UnsupervisedClustering, :DocTestSetup, :(using UnsupervisedClustering); recursive=true)

makedocs(;
    modules=[UnsupervisedClustering],
    authors="raphasampaio, joaquimg, mvpoggi, and vidalt",
    repo="https://github.com/raphasampaio/UnsupervisedClustering.jl/blob/{commit}{path}#{line}",
    sitename="UnsupervisedClustering.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://raphasampaio.github.io/UnsupervisedClustering.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/raphasampaio/UnsupervisedClustering.jl",
    devbranch="main",
)

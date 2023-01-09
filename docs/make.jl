import Pkg
Pkg.instantiate()

using Documenter
using UnsupervisedClustering

DocMeta.setdocmeta!(
    UnsupervisedClustering, 
    :DocTestSetup, 
    :(using UnsupervisedClustering);
    recursive=true
)

makedocs(;
    modules=[UnsupervisedClustering],
    doctest=true,
    clean=true,
    authors="Raphael Araujo Sampaio and Joaquim Dias Garcia and Marcus Poggi and Thibaut Vidal",
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
    repo="github.com/raphasampaio/UnsupervisedClustering.jl.git",
    devbranch="main",
    push_preview = true,
)

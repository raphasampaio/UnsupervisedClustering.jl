using Documenter
using UnsupervisedClustering

DocMeta.setdocmeta!(UnsupervisedClustering, :DocTestSetup, :(using UnsupervisedClustering); recursive = true)

makedocs(
    sitename = "UnsupervisedClustering",
    modules = [UnsupervisedClustering],
    authors = "Raphael Araujo Sampaio and Joaquim Dias Garcia and Marcus Poggi and Thibaut Vidal",
    repo = "https://github.com/raphasampaio/UnsupervisedClustering.jl/blob/{commit}{path}#{line}",
    doctest = true,
    clean = true,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://raphasampaio.github.io/UnsupervisedClustering.jl",
        edit_link = "main",
        assets = [
            "assets/favicon.ico",
        ],
    ),
    pages = [
        "Home" => "index.md"
        "Local Search" => Any[
            "k-means"=>"localsearch/kmeans.md",
            "k-medoids"=>"localsearch/kmedoids.md",
            "GMM"=>"localsearch/gmm.md",
        ]
        "Metaheuristic" => Any[
            "Multi-Start"=>"metaheuristic/multistart.md",
            "Random Swap"=>"metaheuristic/randomswap.md",
            "Genetic Algorithm"=>"metaheuristic/geneticalgorithm.md",
        ]
        "Ensemble" => Any[
            "Clustering Chain"=>"ensemble/chain.md"
        ]
    ],
)

deploydocs(
    repo = "github.com/raphasampaio/UnsupervisedClustering.jl.git",
    devbranch = "main",
    push_preview = true,
)

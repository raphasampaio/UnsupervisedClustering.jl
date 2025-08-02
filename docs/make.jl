using Documenter
using UnsupervisedClustering

DocMeta.setdocmeta!(UnsupervisedClustering, :DocTestSetup, :(using UnsupervisedClustering); recursive = true)

Documenter.makedocs(
    sitename = "UnsupervisedClustering",
    modules = [UnsupervisedClustering],
    authors = "Raphael Araujo Sampaio and Joaquim Dias Garcia and Marcus Poggi and Thibaut Vidal",
    repo = "https://github.com/raphasampaio/UnsupervisedClustering.jl/blob/{commit}{path}#{line}",
    doctest = true,
    clean = true,
    checkdocs = :none,
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
            "Multi-Start"=>"metaheuristic/multi_start.md",
            "Random Swap"=>"metaheuristic/random_swap.md",
            "Genetic Algorithm"=>"metaheuristic/genetic_algorithm.md",
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

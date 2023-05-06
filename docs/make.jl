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
        "Local Searches" => Any[
            "k-means"=>"localsearches/kmeans.md",
            "k-medoids"=>"localsearches/kmedoids.md",
            "GMM"=>"localsearches/gmm.md",
        ]
        "Metaheuristics" => Any[
            "Multi-Start"=>"metaheuristics/multistart.md",
            "Random Swap"=>"metaheuristics/randomswap.md",
            "Genetic Algorithm"=>"metaheuristics/geneticalgorithm.md",
        ]
        "Other" => Any[
            "Clustering Chain"=>"other/chain.md"
        ]
    ],
)

deploydocs(
    repo = "github.com/raphasampaio/UnsupervisedClustering.jl.git",
    devbranch = "main",
    push_preview = true,
)

kmeans_hg(X::Matrix{T}, k::Int) where {T} = _geneticalgorithm(ClusteringData(X, k), _kmeans!)
cmeans_hg(X::Matrix{T}, k::Int) where {T} = _geneticalgorithm(ClusteringData(X, k), _cmeans!)

kellipses_hg(X::Matrix{T}, k::Int) where {T} = _geneticalgorithm(ClusteringData(X, k), _kellipses!)
kellipses_hg_shrunk(X::Matrix{T}, k::Int) where {T} = _geneticalgorithm(ClusteringData(X, k), _kellipses_shrunk!)
kellipses_hg_oas(X::Matrix{T}, k::Int) where {T} = _geneticalgorithm(ClusteringData(X, k), _kellipses_oas!)
kellipses_hg_ledoitwolf(X::Matrix{T}, k::Int) where {T} = _geneticalgorithm(ClusteringData(X, k), _kellipses_ledoitwolf!)

gmm_hg(X::Matrix{T}, k::Int) where {T} = _geneticalgorithm(ClusteringData(X, k), _gmm!)
gmm_hg_shrunk(X::Matrix{T}, k::Int) where {T} = _geneticalgorithm(ClusteringData(X, k), _gmm_shrunk!)
gmm_hg_oas(X::Matrix{T}, k::Int) where {T} = _geneticalgorithm(ClusteringData(X, k), _gmm_oas!)
gmm_hg_ledoitwolf(X::Matrix{T}, k::Int) where {T} = _geneticalgorithm(ClusteringData(X, k), _gmm_ledoitwolf!)

function _geneticalgorithm(data::ClusteringData, method::Function) where {T}
    generation = Generation()

    pi_max = 20
    pi_min = 10

    best_totalcost = 0.0
    iterations_without_improvement = 0

    for i in 1:pi_max
        add_random!(data, generation, method)
    end

    for i in 1:DEFAULT_GLOBAL_ITERATIONS
        parent1, parent2 = binary_tournament(generation)
        child = crossover(data, parent1, parent2)

        # MUTATE
        random_swap!(data, child)

        # LOCAL SEARCH
        method(data, child)
        add!(generation, child)

        size = active_population_size(generation)
        if size > pi_max
            to_remove = size - pi_min
            eliminate(generation, to_remove)
        end

        leader = partialsort(generation.population, 1, lt = isbetter)

        # println("$(leader.totalcost)\t$(adjusted_rand_score(EXPECTED, leader.assignments))")
        if leader.totalcost == best_totalcost
            iterations_without_improvement += 1

            if iterations_without_improvement > DEFAULT_GLOBAL_ITERATIONS
                return leader
            end
        else
            best_totalcost = leader.totalcost
            iterations_without_improvement = 0
        end
    end

    return partialsort(generation.population, 1, lt = isbetter)
end

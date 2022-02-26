kmeans_rs(X::Matrix{T}, k::Int) where T = _randomswap(ClusteringData(X, k), _kmeans!)
cmeans_rs(X::Matrix{T}, k::Int) where T = _randomswap(ClusteringData(X, k), _cmeans!)

kellipses_rs(X::Matrix{T}, k::Int) where T = _randomswap(ClusteringData(X, k), _kellipses!)
kellipses_rs_shrunk(X::Matrix{T}, k::Int) where T = _randomswap(ClusteringData(X, k), _kellipses_shrunk!)
kellipses_rs_oas(X::Matrix{T}, k::Int) where T = _randomswap(ClusteringData(X, k), _kellipses_oas!)
kellipses_rs_ledoitwolf(X::Matrix{T}, k::Int) where T = _randomswap(ClusteringData(X, k), _kellipses_ledoitwolf!)

gmm_rs(X::Matrix{T}, k::Int) where T = _randomswap(ClusteringData(X, k), _gmm!)
gmm_rs_shrunk(X::Matrix{T}, k::Int) where T = _randomswap(ClusteringData(X, k), _gmm_shrunk!)
gmm_rs_oas(X::Matrix{T}, k::Int) where T = _randomswap(ClusteringData(X, k), _gmm_oas!)
gmm_rs_ledoitwolf(X::Matrix{T}, k::Int) where T = _randomswap(ClusteringData(X, k), _gmm_ledoitwolf!)

function _randomswap(data::ClusteringData, method::Function)
    best_result = method(data)

    iterations_without_improvement = 0

    for i in 1:MAX_ITERATIONS
        result = copy(best_result)

        random_swap!(data, result) 
        method(data, result)

        if isbetter(result, best_result)
            best_result = result
            iterations_without_improvement = 0
        else
            iterations_without_improvement += 1
            if iterations_without_improvement > METAHEURISTIC_ITERATIONS
                break
            end
        end
    end
    return best_result
end
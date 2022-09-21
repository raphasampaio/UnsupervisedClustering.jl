kmeans_rs(X::Matrix{T}, k::Int) where {T} = _randomswap(ClusteringData(X, k), _kmeans!)

gmm_rs(X::Matrix{T}, k::Int) where {T} = _randomswap(ClusteringData(X, k), _gmm!)
gmm_rs_shrunk(X::Matrix{T}, k::Int) where {T} = _randomswap(ClusteringData(X, k), _gmm_shrunk!)
gmm_rs_oas(X::Matrix{T}, k::Int) where {T} = _randomswap(ClusteringData(X, k), _gmm_oas!)
gmm_rs_ledoitwolf(X::Matrix{T}, k::Int) where {T} = _randomswap(ClusteringData(X, k), _gmm_ledoitwolf!)

function _randomswap(data::ClusteringData, method::Function)
    best_result = method(data)

    iterations_without_improvement = 0

    for _ in 1:MAX_GLOBAL_ITERATIONS
        result = copy(best_result)

        random_swap!(data, result)
        method(data, result)

        if isbetter(result, best_result)
            best_result = result
            iterations_without_improvement = 0
        else
            iterations_without_improvement += 1
            if iterations_without_improvement > MAX_ITERATIONS_WITHOUT_IMPROVEMENT
                break
            end
        end
    end
    return best_result
end

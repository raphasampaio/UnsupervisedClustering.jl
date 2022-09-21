kmeans_ms(X::Matrix{T}, k::Int) where {T} = _multistart(X, k, kmeans)

gmm_ms(X::Matrix{T}, k::Int) where {T} = _multistart(X, k, gmm)
gmm_ms_shrunk(X::Matrix{T}, k::Int) where {T} = _multistart(X, k, gmm_shrunk)
gmm_ms_oas(X::Matrix{T}, k::Int) where {T} = _multistart(X, k, gmm_oas)
gmm_ms_ledoitwolf(X::Matrix{T}, k::Int) where {T} = _multistart(X, k, gmm_ledoitwolf)

function _multistart(X::Matrix{T}, k::Int, method::Function) where {T}
    best_result = method(X, k)

    for i in 1:MAX_GLOBAL_ITERATIONS
        result = method(X, k)

        if isbetter(result, best_result)
            best_result = result
        end
    end
    return best_result
end

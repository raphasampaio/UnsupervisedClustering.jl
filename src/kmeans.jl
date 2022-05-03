kmeans(X::Matrix{T}, k::Int) where {T} = _kmeans!(ClusteringData(X, k))

function _kmeans!(data::ClusteringData)
    result = HardSphericalResult(data)

    initialize_centers!(data, result)

    _kmeans!(data, result)

    return result
end

function _kmeans!(data::ClusteringData, result::HardSphericalResult)
    n = data.n
    d = data.d
    k = data.k
    max_iterations = data.max_iterations

    centers = zeros(d, k)
    for i in 1:d
        for j in 1:k
            centers[i, j] = result.centers[j][i]
        end
    end

    fitted = Clustering.kmeans!(data.X', centers, maxiter = max_iterations, tol = 1e-3)

    result.assignments = fitted.assignments
    for i in 1:d
        for j in 1:k
            result.centers[j][i] = fitted.centers[i, j]
        end
    end
    result.totalcost = fitted.totalcost

    return nothing
end

kmeans(X::Matrix{T}, k::Int) where T  = _kmeans!(ClusteringData(X, k))

function _kmeans!(data::ClusteringData)
    result = HardSphericalResult(data)

    initialize_centers!(data, result)

    _kmeans!(data, result)

    return result
end

function _kmeans!(data::ClusteringData, result::HardSphericalResult)
    k = data.k

    centers = zeros(data.d, data.k)
    for i in 1:data.d
        for j in 1:data.k
            centers[i, j] = result.centers[j][i]
        end
    end

    fitted = Clustering.kmeans!(data.X', centers, maxiter=MAX_ITERATIONS, tol=1e-3)

    result.assignments = fitted.assignments
    for i in 1:data.d
        for j in 1:data.k
            result.centers[j][i] = fitted.centers[i, j]
        end
    end
    result.totalcost = fitted.totalcost

    # method = KMeans(
    #     n_clusters=k, 
    #     init=centers, 
    #     n_init=1, 
    #     max_iter=max_iterations, 
    #     tol=1e-3, 
    #     precompute_distances=false, 
    #     random_state=1, 
    #     algorithm="full")

    # fitted = fit!(method, data.X)

    # result.assignments = copy(convert(Vector{Int}, predict(fitted, data.X)) .+ 1)
    # for i in 1:data.k
    #     for j in 1:data.d
    #         result.centers[i][j] = fitted[:cluster_centers_][i, j]
    #     end
    # end
    # result.totalcost = fitted[:inertia_]
end
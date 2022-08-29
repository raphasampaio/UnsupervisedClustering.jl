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

    points = zeros(d, n)
    for i in 1:d
        for j in 1:n
            points[i, j] = data.X[j, i]
        end
    end

    centers = zeros(d, k)
    for i in 1:d
        for j in 1:k
            centers[i, j] = result.centers[j][i]
        end
    end

    previous_totalcost = -Inf
    elements = zeros(k)

    for _ in 1:max_iterations
        previous_totalcost = result.totalcost

        # Assignment Step
        result.totalcost = 0
        distances = pairwise(SqEuclidean(), centers, points, dims=2)
        for i in 1:n
            assignment = argmin(distances[:, i])

            result.assignments[i] = assignment
            result.totalcost += distances[assignment, i]
        end

        # Stopping Condition
        if abs(result.totalcost - previous_totalcost) < 1e-3
            break
        end

        # Update Step
        for i in 1:k
            elements[i] = 0
            centers[:, i] .= 0
        end

        for i in 1:n
            assignment = result.assignments[i]

            centers[:, assignment] += points[:, i]
            elements[assignment] += 1
        end
    
        for i in 1:k
            centers[:, i] ./= max(1, elements[i])
        end
    end

    for i in 1:d
        for j in 1:k
            result.centers[j][i] = centers[i, j]
        end
    end

    return result
end

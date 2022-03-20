abstract type Result end

mutable struct HardSphericalResult <: Result
    assignments::Vector{Int}
    centers::Vector{Vector{Float64}}
    totalcost::Float64

    function HardSphericalResult(assignments::Vector{Int}, centers::Vector{Vector{Float64}}, totalcost::Float64)
        return new(assignments, centers, totalcost)
    end

    function HardSphericalResult(data::ClusteringData)
        n = data.n
        d = data.d
        k = data.k

        assignments = zeros(Int, n)
        centers = [zeros(d) for i in 1:k]

        return HardSphericalResult(assignments, centers, Inf)
    end
end

mutable struct SoftSphericalResult <: Result
    assignments::Vector{Int}

    weights::Matrix{Float64}
    centers::Vector{Vector{Float64}}
    totalcost::Float64

    function SoftSphericalResult(assignments::Vector{Int}, weights::Matrix{Float64}, centers::Vector{Vector{Float64}}, totalcost::Float64)
        return new(assignments, weights, centers, totalcost)
    end

    function SoftSphericalResult(data::ClusteringData)
        n = data.n
        d = data.d
        k = data.k

        assignments = zeros(Int, n)
        weights = zeros(n, k)
        centers = [zeros(d) for i in 1:k]

        return SoftSphericalResult(assignments, weights, centers, -Inf)
    end
end

mutable struct HardResult <: Result
    assignments::Vector{Int}
    centers::Vector{Vector{Float64}}
    L::Vector{LowerTriangular{Float64,Matrix{Float64}}}
    totalcost::Float64

    function HardResult(
        assignments::Vector{Int},
        centers::Vector{Vector{Float64}},
        L::Vector{LowerTriangular{Float64,Matrix{Float64}}},
        totalcost::Float64
    )
        return new(assignments, centers, L, totalcost)
    end

    function HardResult(data::ClusteringData)
        n = data.n
        d = data.d
        k = data.k

        assignments = zeros(Int, n)
        centers = [zeros(d) for i in 1:k]
        L = [cholesky(Matrix{Float64}(I, d, d)).L for i in 1:k]

        return HardResult(assignments, centers, L, Inf)
    end
end

mutable struct SoftResult <: Result
    assignments::Vector{Int}

    weights::Vector{Float64}
    centers::Vector{Vector{Float64}}
    covariances::Vector{Matrix{Float64}}

    L::Vector{LowerTriangular{Float64,Matrix{Float64}}}

    totalcost::Float64

    # model::Model

    function SoftResult(
        assignments::Vector{Int},
        weights::Vector{Float64},
        centers::Vector{Vector{Float64}},
        covariances::Vector{Matrix{Float64}},
        L::Vector{LowerTriangular{Float64,Matrix{Float64}}},
        totalcost::Float64
    )#, model::Model)
        return new(assignments, weights, centers, covariances, L, totalcost)#, model)
    end

    function SoftResult(data::ClusteringData)
        n = data.n
        d = data.d
        k = data.k

        assignments = zeros(Int, n)
        weights = zeros(k)
        centers = [zeros(d) for i in 1:k]
        covariances = [Matrix{Float64}(I, d, d) for i in 1:k]
        L = [cholesky(Matrix{Float64}(I, d, d)).L for i in 1:k]

        # model = init_model(d, 1e-2)
        return SoftResult(assignments, weights, centers, covariances, L, -Inf)#, model)
    end
end

function isbetter(a::HardSphericalResult, b::HardSphericalResult)
    return isless(a.totalcost, b.totalcost)
end

function isbetter(a::HardResult, b::HardResult)
    return isless(a.totalcost, b.totalcost)
end

function isbetter(a::SoftSphericalResult, b::SoftSphericalResult)
    return isless(b.totalcost, a.totalcost)
end

function isbetter(a::SoftResult, b::SoftResult)
    return isless(b.totalcost, a.totalcost)
end

function Base.println(result::Result)
    println("totalcost: $(result.totalcost)")
    return nothing
end

function Base.copy(result::HardSphericalResult)
    return HardSphericalResult(copy(result.assignments), deepcopy(result.centers), result.totalcost)
end

function Base.copy(result::HardResult)
    return HardResult(copy(result.assignments), deepcopy(result.centers), deepcopy(result.L), result.totalcost)
end

function Base.copy(result::SoftSphericalResult)
    return SoftSphericalResult(copy(result.assignments), copy(result.weights), deepcopy(result.centers), result.totalcost)
end

function Base.copy(result::SoftResult)
    return SoftResult(
        copy(result.assignments),
        copy(result.weights),
        deepcopy(result.centers),
        deepcopy(result.covariances),
        deepcopy(result.L),
        result.totalcost
    )#, result.model)
end

function distance(a::HardSphericalResult, i::Int, b::HardSphericalResult, j::Int)
    return Distances.evaluate(Euclidean(), a.centers[i], b.centers[j])
end

function distance(a::SoftSphericalResult, i::Int, b::SoftSphericalResult, j::Int)
    return Distances.evaluate(Euclidean(), a.centers[i], b.centers[j])
end

function distance(a::HardResult, i::Int, b::HardResult, j::Int)
    # return Distances.evaluate(Euclidean(), a.centers[i], b.centers[j])
    distance1 = Distances.sqmahalanobis(a.centers[i], b.centers[j], a.L[i])
    distance2 = Distances.sqmahalanobis(a.centers[i], b.centers[j], b.L[j])
    return (distance1 + distance2) / 2
end

function distance(a::SoftResult, i::Int, b::SoftResult, j::Int)
    # return Distances.evaluate(Euclidean(), a.centers[i], b.centers[j])
    distance1 = Distances.sqmahalanobis(a.centers[i], b.centers[j], a.L[i])
    distance2 = Distances.sqmahalanobis(a.centers[i], b.centers[j], b.L[j])
    return (distance1 + distance2) / 2
end

function random_swap!(data::ClusteringData, result::HardSphericalResult)
    n = data.n
    d = data.d
    k = data.k

    i = rand(1:k)
    result.centers[i] = copy(data.X[rand(1:n), :])
    invalidate!(result)
    return nothing
end

function random_swap!(data::ClusteringData, result::SoftSphericalResult)
    n = data.n
    d = data.d
    k = data.k

    i = rand(1:k)
    result.centers[i] = copy(data.X[rand(1:n), :])
    # result.weights .= 1 / n
    invalidate!(result)
    return nothing
end

function random_swap!(data::ClusteringData, result::HardResult)
    n = data.n
    d = data.d
    k = data.k

    i = rand(1:k)

    result.centers[i] = copy(data.X[rand(1:n), :])

    size = mean([det(result.L[j]) for j in 1:k])
    value = (size > 0 ? size : 1.0)^(1 / d)
    result.L[i] = cholesky(value .* Matrix{Float64}(I, d, d)).L
    invalidate!(result)
    return nothing
end

function random_swap!(data::ClusteringData, result::SoftResult)
    n = data.n
    d = data.d
    k = data.k

    i = rand(1:k)

    result.centers[i] = copy(data.X[rand(1:n), :])

    size = mean([det(result.covariances[j]) for j in 1:k])
    value = (size > 0 ? size : 1.0)^(1 / d)
    result.covariances[i] = value .* Matrix{Float64}(I, d, d)
    result.L[i] = cholesky(result.covariances[i]).L

    # method2
    # swapped_matrix = zeros(d, d)
    # for j in 1:k
    #     # if i != j
    #         for l1 in 1:d
    #             for l2 in 1:d
    #                 swapped_matrix[l1, l2] += result.covariances[j][l1, l2]
    #             end
    #         end
    #     # end
    # end
    # for l1 in 1:d
    #     for l2 in 1:d
    #         # swapped_matrix[l1, l2] /= (k-1)
    #         swapped_matrix[l1, l2] /= k
    #     end
    # end
    # result.covariances[i] = Hermitian(swapped_matrix)

    # COVARIANCES
    # cache = zeros(d, d)
    # for i in 1:k
    #     cache .+= result.covariances[i]
    # end
    # result.covariances[i] = Hermitian(cache ./ k)
    # result.L[i] = cholesky(result.covariances[i]).L

    # WEIGHTS
    # result.weights .= 1 / n

    invalidate!(result)
    return nothing
end

function copy_clusters!(destiny::HardSphericalResult, destiny_i::Int, source::HardSphericalResult, source_i::Int)
    destiny.centers[destiny_i] = copy(source.centers[source_i])
    return nothing
end

function copy_clusters!(destiny::SoftSphericalResult, destiny_i::Int, source::SoftSphericalResult, source_i::Int)
    destiny.centers[destiny_i] = copy(source.centers[source_i])
    return nothing
end

function copy_clusters!(destiny::HardResult, destiny_i::Int, source::HardResult, source_i::Int)
    destiny.centers[destiny_i] = copy(source.centers[source_i])
    destiny.L[destiny_i] = copy(source.L[source_i])
    return nothing
end

function copy_clusters!(destiny::SoftResult, destiny_i::Int, source::SoftResult, source_i::Int)
    destiny.centers[destiny_i] = copy(source.centers[source_i])
    destiny.covariances[destiny_i] = copy(source.covariances[source_i])
    destiny.L[destiny_i] = copy(source.L[source_i])
    return nothing
end

function invalidate!(result::HardSphericalResult)
    result.assignments .= -1
    result.totalcost = Inf
    return nothing
end

function invalidate!(result::SoftSphericalResult)
    result.assignments .= -1
    result.totalcost = -Inf
    return nothing
end

function invalidate!(result::HardResult)
    result.assignments .= -1
    result.totalcost = Inf
    return nothing
end

function invalidate!(result::SoftResult)
    result.assignments .= -1
    result.totalcost = -Inf
    return nothing
end

function initialize_centers!(data::ClusteringData, result::Result)
    X = data.X

    index = randperm(data.n)
    for i in 1:data.k
        for j in 1:data.d
            result.centers[i][j] = X[index[i], j]
        end
    end
    return nothing
end

# global ITERATOR = 1

# function Base.show(io::IO, result::SoftResult)
# ari = round.(adjusted_rand_score(EXPECTED, result.assignments), digits=2)
# obj = round.(result.totalcost, digits=2)
# print(io, "ari: $ari, obj: $obj [$ITERATOR]")

# assignments = result.assignments
# @rput assignments

# @rput ITERATOR
# R"""
#     print(ITERATOR)
#     data <- read.csv("D:\\Dropbox (PSR)\\Clustering Datasets\\others\\htru2.csv", header=FALSE, stringsAsFactors=FALSE)
#     png(sprintf("D:\\Development\\clustering\\img\\uci\\%d.png", ITERATOR), height = 900, width = 900)
#     g <- ggpairs(data, columns = 2:ncol(data), ggplot2::aes(colour=as.factor(assignments)))
#     print(g)
#     dev.off()
# """

# global ITERATOR += 1
# end

function update_weights!(child::HardSphericalResult, parent1::HardSphericalResult, parent2::HardSphericalResult, assignment::Vector{Int}) end

function update_weights!(child::SoftSphericalResult, parent1::SoftSphericalResult, parent2::SoftSphericalResult, assignment::Vector{Int}) end

function update_weights!(child::HardResult, parent1::HardResult, parent2::HardResult, assignment::Vector{Int}) end

function update_weights!(child::SoftResult, parent1::SoftResult, parent2::SoftResult, assignment::Vector{Int})
    k = length(assignment)
    for i in 1:k
        child.weights[i] = (parent1.weights[i] + parent2.weights[assignment[i]]) / 2
    end
    return nothing
end

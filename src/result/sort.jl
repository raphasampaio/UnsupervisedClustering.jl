function Base.sort!(
    result::KmeansResult;
    lt = Base.isless,
    data::AbstractArray{<:Number} = result.objective_per_cluster,
    rev::Bool = false,
)
    k = result.k
    n = length(result.assignments)

    permutation = sortperm(data, lt = lt, rev = rev)

    map = zeros(Int, k)
    for i in 1:k
        map[permutation[i]] = i
    end

    for i in 1:n
        result.assignments[i] = map[result.assignments[i]]
    end

    permutecols!(result.clusters, copy(permutation))

    permute!(result.objective_per_cluster, permutation)

    return nothing
end

function Base.sort!(
    result::KmedoidsResult;
    lt = Base.isless,
    data::AbstractArray{<:Number} = result.objective_per_cluster,
    rev::Bool = false,
)
    k = result.k
    n = length(result.assignments)

    permutation = sortperm(data, lt = lt, rev = rev)

    map = zeros(Int, k)
    for i in 1:k
        map[permutation[i]] = i
    end

    for i in 1:n
        result.assignments[i] = map[result.assignments[i]]
    end

    permute!(result.clusters, permutation)

    permute!(result.objective_per_cluster, permutation)

    return nothing
end

# function Base.sort!(result::GMMResult)
# end

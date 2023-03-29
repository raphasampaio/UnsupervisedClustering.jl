function Base.copy(result::KmeansResult)
    return KmeansResult(
        result.k,
        copy(result.assignments),
        copy(result.centers),
        result.objective,
        copy(result.objective_per_cluster),
        result.iterations,
        result.elapsed,
        result.converged
    )
end

function Base.copy(result::KmedoidsResult)
    return KmedoidsResult(
        result.k,
        copy(result.assignments),
        copy(result.centers),
        result.objective,
        copy(result.objective_per_cluster),
        result.iterations,
        result.elapsed,
        result.converged
    )
end

function Base.copy(result::GMMResult)
    return GMMResult(
        result.k,
        copy(result.assignments),
        copy(result.weights),
        deepcopy(result.centers),
        deepcopy(result.covariances),
        result.objective,
        result.iterations,
        result.elapsed,
        result.converged
    )
end

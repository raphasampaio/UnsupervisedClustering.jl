function counts(result::ClusteringResult)
    return StatsBase.counts(result.assignments, result.k)
end

function Base.copy(result::KmeansResult)
    return KmeansResult(
        result.k,
        copy(result.assignments),
        copy(result.centers),
        result.objective,
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

function concatenate(results::KmeansResult...)
    size = length(results)

    accumulated_k = zeros(size)
    for i in 1:size
        if i > 1
            accumulated_k[i] = accumulated_k[i - 1] + results[i - 1].k
        end
    end

    k = sum([result.k for result in results])
    assignments = vcat([results[i].assignments .+ accumulated_k[i] for i in 1:size]...)
    centers = hcat([results[i].centers for i in 1:size]...)
    objective = sum([result.objective for result in results])
    iterations = sum([result.iterations for result in results])
    elapsed = sum([result.elapsed for result in results])
    converged = all([result.converged for result in results])

    return KmeansResult(k, assignments, centers, objective, iterations, elapsed, converged)
end
# function concatenate_k(result::Result, results::Result...)
#     return result.k + sum([i.k for i in results])
# end

function concatenate_assignments(result::Result, results::Result...)
    size = length(results)
    accumulated = zeros(Int, size)
    for i in 1:size
        if i == 1
            accumulated[i] = result.k
        else
            accumulated[i] = accumulated[i-1] + results[i-1].k
        end
    end
    return vcat(result.assignments, [results[i].assignments .+ accumulated[i] for i in 1:size]...)
end

function concatenate_clusters(result::KmeansResult, results::KmeansResult...)
    return hcat(result.clusters, [i.clusters for i in results]...)
end

function concatenate_clusters(result::KmedoidsResult, results::KmedoidsResult...)
    size = length(results)
    accumulated = zeros(Int, size)
    for i in 1:size
        if i == 1
            accumulated[i] = length(result.assignments)
        else
            accumulated[i] = accumulated[i-1] + length(results[i-1].assignments)
        end
    end
    return vcat(result.clusters, [results[i].clusters .+ accumulated[i] for i in 1:size]...)
end

function concatenate_objective(result::Result, results::Result...)
    return result.objective + sum([i.objective for i in results])
end

function concatenate_objective_per_cluster(result::Result, results::Result...)
    return vcat(result.objective_per_cluster, [i.objective_per_cluster for i in results]...)
end

function concatenate_iterations(result::Result, results::Result...)
    return result.iterations + sum([i.iterations for i in results])
end

function concatenate_elapsed(result::Result, results::Result...)
    return result.elapsed + sum([i.elapsed for i in results])
end

function concatenate_converged(result::Result, results::Result...)
    return result.converged && all([i.converged for i in results])
end

function concatenate(result::KmeansResult, results::KmeansResult...)::KmeansResult
    assignments = concatenate_assignments(result, results...)
    clusters = concatenate_clusters(result, results...)
    objective = concatenate_objective(result, results...)
    objective_per_cluster = concatenate_objective_per_cluster(result, results...)
    iterations = concatenate_iterations(result, results...)
    elapsed = concatenate_elapsed(result, results...)
    converged = concatenate_converged(result, results...)
    return KmeansResult(assignments, clusters, objective, objective_per_cluster, iterations, elapsed, converged)
end

function concatenate(result::KmedoidsResult, results::KmedoidsResult...)::KmedoidsResult
    assignments = concatenate_assignments(result, results...)
    clusters = concatenate_clusters(result, results...)
    objective = concatenate_objective(result, results...)
    objective_per_cluster = concatenate_objective_per_cluster(result, results...)
    iterations = concatenate_iterations(result, results...)
    elapsed = concatenate_elapsed(result, results...)
    converged = concatenate_converged(result, results...)
    return KmedoidsResult(assignments, clusters, objective, objective_per_cluster, iterations, elapsed, converged)
end

# function concatenate(result::GMMResult, results::GMMResult...)::GMMResult
# end

function concatenate_k(result::ClusteringResult, results::ClusteringResult...)
    return result.k + sum([i.k for i in results])
end

function concatenate_assignments(result::ClusteringResult, results::ClusteringResult...)
    size = length(results)
    accumulated = zeros(size)
    for i in 1:size
        if i == 1
            accumulated[i] = result.k
        else
            accumulated[i] = accumulated[i - 1] + results[i - 1].k
        end
    end
    return vcat(result.assignments, [results[i].assignments .+ accumulated[i] for i in 1:size]...)
end

function concatenate_centers(result::KmeansResult, results::KmeansResult...)
    return hcat(result.centers, [i.centers for i in results]...)
end

function concatenate_centers(result::KmedoidsResult, results::KmedoidsResult...)
    size = length(results)
    accumulated = zeros(size)
    for i in 1:size
        if i == 1
            accumulated[i] = length(result.assignments)
        else
            accumulated[i] = accumulated[i - 1] + length(results[i - 1].assignments)
        end
    end
    return vcat(result.centers, [results[i].centers .+ accumulated[i] for i in 1:size]...)
end

function concatenate_objective(result::ClusteringResult, results::ClusteringResult...)
    return result.objective + sum([i.objective for i in results])
end

function concatenate_iterations(result::ClusteringResult, results::ClusteringResult...)
    return result.iterations + sum([i.iterations for i in results])
end

function concatenate_elapsed(result::ClusteringResult, results::ClusteringResult...)
    return result.elapsed + sum([i.elapsed for i in results])
end

function concatenate_converged(result::ClusteringResult, results::ClusteringResult...)
    return result.converged && all([i.converged for i in results])
end

function concatenate(result::KmeansResult, results::KmeansResult...)::KmeansResult
    k = concatenate_k(result, results...)
    assignments = concatenate_assignments(result, results...)
    centers = concatenate_centers(result, results...)
    objective = concatenate_objective(result, results...)
    iterations = concatenate_iterations(result, results...)
    elapsed = concatenate_elapsed(result, results...)
    converged = concatenate_converged(result, results...)
    return KmeansResult(k, assignments, centers, objective, iterations, elapsed, converged)
end

function concatenate(result::KmedoidsResult, results::KmedoidsResult...)::KmedoidsResult
    k = concatenate_k(result, results...)
    assignments = concatenate_assignments(result, results...)
    centers = concatenate_centers(result, results...)
    objective = concatenate_objective(result, results...)
    iterations = concatenate_iterations(result, results...)
    elapsed = concatenate_elapsed(result, results...)
    converged = concatenate_converged(result, results...)
    return KmedoidsResult(k, assignments, centers, objective, iterations, elapsed, converged)
end
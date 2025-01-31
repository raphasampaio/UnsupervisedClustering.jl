function reset_objective!(result::KmeansResult)
    result.objective = Inf
    for i in 1:result.k
        result.objective_per_cluster[i] = Inf
    end

    return nothing
end

function reset_objective!(result::KmedoidsResult)
    result.objective = Inf
    for i in 1:result.k
        result.objective_per_cluster[i] = Inf
    end
    return nothing
end

function reset_objective!(result::GMMResult)
    result.objective = -Inf
    return nothing
end

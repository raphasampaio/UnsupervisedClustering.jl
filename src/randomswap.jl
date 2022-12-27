Base.@kwdef struct RandomSwap <: Algorithm
    local_search::Algorithm
    verbose::Bool = false
    max_iterations::Integer = 200
    max_iterations_without_improvement::Integer = 150
end

function train(parameters::RandomSwap, data::AbstractMatrix{<:Real}, k::Integer)::Result
    best_result = train(parameters.local_search, data, k)

    iterations_without_improvement = 0

    for iteration in 1:parameters.max_iterations
        result = copy(best_result)

        random_swap!(result, data, parameters.local_search.rng)
        train!(parameters.local_search, data, result)

        if isbetter(result, best_result)
            best_result = result
            iterations_without_improvement = 0
        else
            iterations_without_improvement += 1
            if iterations_without_improvement > parameters.max_iterations_without_improvement
                break
            end
        end

        if parameters.verbose
            println("$iteration - $(best_result.objective) ($iterations_without_improvement)")
        end
    end

    return best_result
end

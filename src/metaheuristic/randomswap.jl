@doc raw"""
    RandomSwap(
        local_search::ClusteringAlgorithm
        verbose::Bool = DEFAULT_VERBOSE
        max_iterations::Integer = 200
        max_iterations_without_improvement::Integer = 150
    )

TODO: Documentation
"""
Base.@kwdef struct RandomSwap <: ClusteringAlgorithm
    local_search::ClusteringAlgorithm
    verbose::Bool = DEFAULT_VERBOSE
    max_iterations::Integer = 200
    max_iterations_without_improvement::Integer = 150
end

@doc raw"""
    fit(meta::RandomSwap, data::AbstractMatrix{<:Real}, k::Integer)

TODO: Documentation
"""
function fit(meta::RandomSwap, data::AbstractMatrix{<:Real}, k::Integer)::ClusteringResult
    iterations_without_improvement = 0

    best_result = fit(meta.local_search, data, k)

    if meta.verbose
        print_iteration(0)
        print_iteration(iterations_without_improvement)
        print_result(best_result)
        print_string("(initial solution)")
        print_newline()
    end

    for iteration in 1:meta.max_iterations
        result = copy(best_result)

        random_swap!(result, data, meta.local_search.rng)

        fit!(meta.local_search, data, result)

        if meta.verbose
            print_iteration(iteration)
            print_iteration(iterations_without_improvement)
            print_result(result)
        end

        if isbetter(result, best_result)
            best_result = result
            iterations_without_improvement = 0

            if meta.verbose
                print_string("(new best)")
            end
        else
            iterations_without_improvement += 1
            if iterations_without_improvement > meta.max_iterations_without_improvement
                break
            end
        end

        if meta.verbose
            print_newline()
        end
    end

    return best_result
end

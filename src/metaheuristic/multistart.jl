@doc raw"""
    MultiStart(
        local_search::ClusteringAlgorithm
        verbose::Bool = DEFAULT_VERBOSE
        max_iterations::Integer = 200
    )

The MultiStart approach repeatedly applies a clustering algorithm to generate multiple solutions with different initial points and selects the best solution.

# Fields
- `local_search`: the clustering algorithm applied to improve the solution in each meta-heuristics iteration.
- `verbose`: controls whether the algorithm should display additional information during execution.
- `max_iterations`: represents the maximum number of iterations the algorithm will perform before stopping.
"""
Base.@kwdef struct MultiStart <: ClusteringAlgorithm
    local_search::ClusteringAlgorithm
    verbose::Bool = DEFAULT_VERBOSE
    max_iterations::Integer = 200
end

@doc raw"""
    fit(meta::MultiStart, data::AbstractMatrix{<:Real}, k::Integer)

TODO: Documentation
"""
function fit(meta::MultiStart, data::AbstractMatrix{<:Real}, k::Integer)::ClusteringResult
    best_result = fit(meta.local_search, data, k)

    if meta.verbose
        print_iteration(0)
        print_result(best_result)
        print_string("(initial solution)")
        print_newline()
    end

    for iteration in 1:meta.max_iterations
        result = fit(meta.local_search, data, k)

        if meta.verbose
            print_iteration(iteration)
            print_result(result)
        end

        if isbetter(result, best_result)
            best_result = result

            if meta.verbose
                print_string("(new best)")
            end
        end

        if meta.verbose
            print_newline()
        end
    end

    return best_result
end

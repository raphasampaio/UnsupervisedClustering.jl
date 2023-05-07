@doc """
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

@doc """
    fit(
        meta::MultiStart,
        data::AbstractMatrix{<:Real},
        k::Integer
    )

The `fit` function applies a multi-start to a clustering problem and returns a result object representing the clustering outcome.

# Parameters:
- `meta`: an instance representing the clustering settings and parameters.
- `data`: a floating-point matrix, where each row represents a data point, and each column represents a feature.
- `k`: an integer representing the number of clusters.

# Example

```julia
n = 100
d = 2
k = 2

data = rand(n, d)

kmeans = Kmeans()
multi_start = MultiStart(local_search = kmeans)
result = fit(multi_start, data, k)
```
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

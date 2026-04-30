@doc """
    RandomSwap{LS}(
        local_search::LS
        verbose::Bool = DEFAULT_VERBOSE
        max_iterations::Int = 200
        max_iterations_without_improvement::Int = 150
        discard_empty_clusters::Bool = false
    ) where {LS <: AbstractAlgorithm}

RandomSwap is a meta-heuristic approach used for clustering problems. It follows an iterative process that combines local optimization with perturbation to explore the search space effectively. A local optimization algorithm is applied at each iteration to converge toward a local optimum. Then, a perturbation operator generates a new starting point and continues the search.

# Type Parameters
- `LS`: the specific type of the local search algorithm

# Fields
- `local_search`: the clustering algorithm applied to improve the solution in each meta-heuristics iteration.
- `verbose`: controls whether the algorithm should display additional information during execution.
- `max_iterations`: represents the maximum number of iterations the algorithm will perform before stopping.
- `max_iterations_without_improvement`: represents the maximum number of iterations allowed without improving the best solution.
- `discard_empty_clusters`: when `true`, candidate solutions whose local-search result contains at least one empty cluster are rejected and never replace the current best solution.

# References
* Fränti, Pasi.
  Efficiency of random swap clustering.
  Journal of big data 5.1 (2018): 1-29.
"""
Base.@kwdef struct RandomSwap{LS <: AbstractAlgorithm} <: AbstractAlgorithm
    local_search::LS
    verbose::Bool = DEFAULT_VERBOSE
    max_iterations::Int = 200
    max_iterations_without_improvement::Int = 150
    discard_empty_clusters::Bool = false
end

@doc """
    fit(
        meta::RandomSwap,
        data::AbstractMatrix{<:Real},
        k::Integer
    )

The `fit` function applies a random swap to a clustering problem and returns a result object representing the clustering outcome.

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
random_swap = RandomSwap(local_search = kmeans)
result = fit(random_swap, data, k)
```
"""
function fit(meta::RandomSwap{LS}, data::AbstractMatrix{<:Real}, k::Integer) where {LS <: AbstractAlgorithm}
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

        discard = meta.discard_empty_clusters && has_empty_clusters(result)

        if !discard && isbetter(result, best_result)
            best_result = result
            iterations_without_improvement = 0

            if meta.verbose
                print_string("(new best)")
            end
        else
            iterations_without_improvement += 1

            if meta.verbose && discard
                print_string("(discarded: empty cluster)")
            end

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

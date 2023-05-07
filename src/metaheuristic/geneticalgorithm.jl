@doc """
    GeneticAlgorithm(
        local_search::ClusteringAlgorithm
        verbose::Bool = DEFAULT_VERBOSE
        max_iterations::Integer = 200
        max_iterations_without_improvement::Integer = 150
        π_min::Integer = 40
        π_max::Integer = 50
    )

GeneticAlgorithm represents a clustering algorithm that utilizes a genetic algorithm approach to optimize cluster assignments. It combines evolutionary computation and local search elements to find high-quality clustering solutions.

# Fields
- `local_search`: the clustering algorithm applied to improve the solution in each meta-heuristics iteration.
- `verbose`: controls whether the algorithm should display additional information during execution.
- `max_iterations`: represents the maximum number of iterations the algorithm will perform before stopping.
- `max_iterations_without_improvement`: represents the maximum number of iterations allowed without improving the best solution.
- `π_max`: the maximum population size used in the genetic algorithm.
- `π_min`: the minimum population size used in the genetic algorithm.
"""
Base.@kwdef struct GeneticAlgorithm <: ClusteringAlgorithm
    local_search::ClusteringAlgorithm
    verbose::Bool = DEFAULT_VERBOSE
    max_iterations::Integer = 200
    max_iterations_without_improvement::Integer = 150
    π_min::Integer = 40
    π_max::Integer = 50
end

@doc """
    fit(
        meta::GeneticAlgorithm,
        data::AbstractMatrix{<:Real},
        k::Integer
    )

The `fit` function applies a genetic algorithm to a clustering problem and returns a result object representing the clustering outcome.

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
genetic_algorithm = GeneticAlgorithm(local_search = kmeans)
result = fit(genetic_algorithm, data, k)
```
"""
function fit(meta::GeneticAlgorithm, data::AbstractMatrix{<:Real}, k::Integer)::ClusteringResult
    generation = Generation()

    best_objective = 0.0
    iterations_without_improvement = 0

    for _ in 1:meta.π_max
        result = fit(meta.local_search, data, k)
        add!(generation, result)

        if meta.verbose
            print_iteration(0)
            print_result(result)
            print_string("(initial population)")
            print_newline()
        end
    end

    for iteration in 1:meta.max_iterations
        # PARENTS SELECTION
        parent1, parent2 = binary_tournament(generation, meta.local_search.rng)

        # CROSSOVER
        child = crossover(parent1, parent2, data, meta.local_search.rng)

        # MUTATE
        random_swap!(child, data, meta.local_search.rng)

        # LOCAL SEARCH
        fit!(meta.local_search, data, child)
        add!(generation, child)

        if meta.verbose
            print_iteration(iteration)
            print_iteration(iterations_without_improvement)
            print_result(child)
            print_newline()
        end

        size = active_population_size(generation)
        if size > meta.π_max
            to_remove = size - meta.π_min
            eliminate(generation, to_remove, meta.local_search.rng)
        end

        best_solution = get_best_solution(generation)

        if best_solution.objective ≈ best_objective
            iterations_without_improvement += 1

            if iterations_without_improvement > meta.max_iterations_without_improvement
                return best_solution
            end
        else
            best_objective = best_solution.objective
            iterations_without_improvement = 0
        end
    end

    return get_best_solution(generation)
end

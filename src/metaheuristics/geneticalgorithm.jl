@doc raw"""
    fit(meta::GeneticAlgorithm, data::AbstractMatrix{<:Real}, k::Integer)

TODO: Documentation
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

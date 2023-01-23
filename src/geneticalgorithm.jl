function fit(parameters::GeneticAlgorithm, data::AbstractMatrix{<:Real}, k::Integer)::ClusteringResult
    generation = Generation()

    best_objective = 0.0
    iterations_without_improvement = 0

    for _ in 1:parameters.π_max
        result = fit(parameters.local_search, data, k)
        add!(generation, result)

        if parameters.verbose
            print_iteration(0)
            print_result(result)
            print_string("(initial population)")
            print_newline()
        end
    end

    for iteration in 1:parameters.max_iterations
        # PARENTS SELECTION
        parent1, parent2 = binary_tournament(generation, parameters.local_search.rng)

        # CROSSOVER
        child = crossover(parent1, parent2, data, parameters.local_search.rng)

        # MUTATE
        random_swap!(child, data, parameters.local_search.rng)

        # LOCAL SEARCH
        fit!(parameters.local_search, data, child)
        add!(generation, child)

        if parameters.verbose
            print_iteration(iteration)
            print_iteration(iterations_without_improvement)
            print_result(child)
            print_newline()
        end

        size = active_population_size(generation)
        if size > parameters.π_max
            to_remove = size - parameters.π_min
            eliminate(generation, to_remove, parameters.local_search.rng)
        end

        best_solution = get_best_solution(generation)

        if best_solution.objective ≈ best_objective
            iterations_without_improvement += 1

            if iterations_without_improvement > parameters.max_iterations_without_improvement
                return best_solution
            end
        else
            best_objective = best_solution.objective
            iterations_without_improvement = 0
        end
    end

    return get_best_solution(generation)
end

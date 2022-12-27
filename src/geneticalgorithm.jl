Base.@kwdef struct GeneticAlgorithm <: Algorithm
    local_search::Algorithm
    verbose::Bool = false
    max_iterations::Integer = 200
    max_iterations_without_improvement::Integer = 150
    π_max::Integer = 50
    π_min::Integer = 40
end

function train(parameters::GeneticAlgorithm, data::AbstractMatrix{<:Real}, k::Integer)::Result
    generation = Generation()

    best_objective = 0.0
    iterations_without_improvement = 0

    for _ in 1:parameters.π_max
        add_random!(generation, parameters.local_search, data, k)
    end

    for iteration in 1:parameters.max_iterations
        # PARENTS SELECTION
        parent1, parent2 = binary_tournament(generation, parameters.local_search.rng)

        # CROSSOVER
        child = crossover(parent1, parent2, parameters.local_search.rng)

        # MUTATE
        random_swap!(child, data, parameters.local_search.rng)

        # LOCAL SEARCH
        train!(parameters.local_search, data, child)

        add!(generation, child)

        size = active_population_size(generation)
        if size > parameters.π_max
            to_remove = size - parameters.π_min
            eliminate(generation, to_remove, parameters.local_search.rng)
        end

        leader = partialsort(generation.population, 1, lt = isbetter)

        if leader.objective ≈ best_objective
            iterations_without_improvement += 1

            if iterations_without_improvement > parameters.max_iterations_without_improvement
                return leader
            end
        else
            best_objective = leader.objective
            iterations_without_improvement = 0
        end

        if parameters.verbose
            println("Iteration $iteration - $(leader.objective) ($iterations_without_improvement)")
        end
    end

    return partialsort(generation.population, 1, lt = isbetter)
end
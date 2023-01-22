Base.@kwdef mutable struct RandomSwap <: ClusteringAlgorithm
    const local_search::ClusteringAlgorithm
    verbose::Bool = false
    max_iterations::Integer = 200
    max_iterations_without_improvement::Integer = 150
end

function seed!(algorithm::RandomSwap, seed::Integer)
    Random.seed!(algorithm.local_search.rng, seed)
    return nothing
end

function fit(parameters::RandomSwap, data::AbstractMatrix{<:Real}, k::Integer)::ClusteringResult
    iterations_without_improvement = 0

    best_result = fit(parameters.local_search, data, k)

    if parameters.verbose
        print_iteration(0)
        print_iteration(iterations_without_improvement)
        print_result(best_result)
        print_string("(initial solution)")
        print_newline()
    end

    for iteration in 1:parameters.max_iterations
        result = copy(best_result)

        random_swap!(result, data, parameters.local_search.rng)

        fit!(parameters.local_search, data, result)

        if parameters.verbose
            print_iteration(iteration)
            print_iteration(iterations_without_improvement)
            print_result(result)
        end

        if isbetter(result, best_result)
            best_result = result
            iterations_without_improvement = 0

            if parameters.verbose
                print_string("(new best)")
            end
        else
            iterations_without_improvement += 1
            if iterations_without_improvement > parameters.max_iterations_without_improvement
                break
            end
        end

        if parameters.verbose
            print_newline()
        end
    end

    return best_result
end

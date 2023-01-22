Base.@kwdef mutable struct MultiStart <: ClusteringAlgorithm
    local_search::ClusteringAlgorithm
    verbose::Bool = false
    max_iterations::Integer = 200
end

function seed!(algorithm::MultiStart, seed::Integer)
    Random.seed!(algorithm.local_search.rng, seed)
    return nothing
end

function fit(parameters::MultiStart, data::AbstractMatrix{<:Real}, k::Integer)::ClusteringResult
    best_result = fit(parameters.local_search, data, k)

    if parameters.verbose
        print_iteration(0)
        print_result(best_result)
        print_string("(initial solution)")
        print_newline()
    end

    for iteration in 1:parameters.max_iterations
        result = fit(parameters.local_search, data, k)

        if parameters.verbose
            print_iteration(iteration)
            print_result(result)
        end

        if isbetter(result, best_result)
            best_result = result

            if parameters.verbose
                print_string("(new best)")
            end
        end

        if parameters.verbose
            print_newline()
        end
    end

    return best_result
end

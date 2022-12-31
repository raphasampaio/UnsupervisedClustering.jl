Base.@kwdef struct MultiStart <: Algorithm
    local_search::Algorithm
    verbose::Bool = false
    max_iterations::Integer = 200
end

function train(parameters::MultiStart, data::AbstractMatrix{<:Real}, k::Integer)::Result
    best_result = train(parameters.local_search, data, k)

    if parameters.verbose
        print_iteration(0)
        print_result(best_result)
        print_string("(initial solution)")
        print_newline()
    end

    for iteration in 1:parameters.max_iterations
        result = train(parameters.local_search, data, k)

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

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

Base.@kwdef struct MultiStart <: Algorithm
    local_search::Algorithm
    verbose::Bool = false
    max_iterations::Integer = 200
end

function train(parameters::MultiStart, data::AbstractMatrix{<:Real}, k::Integer)::Result
    best_result = train(parameters.local_search, data, k)

    for iteration in 1:parameters.max_iterations
        result = train(parameters.local_search, data, k)

        if isbetter(result, best_result)
            best_result = result
        end

        if parameters.verbose
            println("$iteration - $(best_result.objective)")
        end
    end

    return best_result
end

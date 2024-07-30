function stochastic_matrix(k::Integer, from::AbstractVector{<:Integer}, to::AbstractVector{<:Integer})
    @assert length(from) == length(to)

    matrix = zeros(Float64, k, k)
    for (i, j) in zip(from, to)
        matrix[i, j] += 1.0
    end
    matrix = matrix ./ sum(matrix, dims = 1)

    @assert !any(isnan, matrix)

    return matrix
end

function stochastic_matrix(from::AbstractResult, to::AbstractResult)
    @assert from.k == to.k
    return stochastic_matrix(from.k, from.assignments, to.assignments)
end

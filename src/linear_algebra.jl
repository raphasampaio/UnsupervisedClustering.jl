function identity_matrix(d::Integer)
    return Symmetric(Matrix{Float64}(I, d, d))
end

function permutecols!(a::AbstractMatrix, p::AbstractVector{<:Integer})
    return Base.permutecols!!(a, copy(p))
end

function build_capacities_vector(n::Integer, k::Integer)
    base = div(n, k)
    remainder = n % k
    capacities = fill(base, k)

    for i in 1:remainder
        capacities[i] += 1
    end

    return capacities
end

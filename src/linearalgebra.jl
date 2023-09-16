function identity_matrix(d::Integer)
    return Matrix{Float64}(I, d, d)
end

function permutecols!(a::AbstractMatrix, p::AbstractVector{<:Integer})
    return Base.permutecols!!(a, copy(p))
end

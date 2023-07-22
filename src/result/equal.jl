function Base.:(==)(a::KmeansResult, b::KmeansResult)
    return a.k == b.k && a.assignments == b.assignments && a.clusters == b.clusters
end

# function Base.:(==)(a::KmedoidsResult, b::KmedoidsResult)
#     return a.k == b.k && a.assignments == b.assignments && a.clusters == b.clusters
# end

function Base.:(==)(a::GMMResult, b::GMMResult)
    return a.k == b.k && a.assignments == b.assignments && a.weights == b.weights && a.clusters == b.clusters && a.covariances == b.covariances
end

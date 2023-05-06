@doc raw"""
    ClusteringChain(
        algorithms::Vector{ClusteringAlgorithm}
    )

TODO: Documentation
"""
Base.@kwdef struct ClusteringChain <: ClusteringAlgorithm
    algorithms::Vector{ClusteringAlgorithm}
end

function ClusteringChain(algorithms::ClusteringAlgorithm...)
    return ClusteringChain(collect(algorithms))
end

@doc raw"""
    fit(chain::ClusteringChain, data::AbstractMatrix{<:Real}, k::Integer)

TODO: Documentation
"""
function fit(chain::ClusteringChain, data::AbstractMatrix{<:Real}, k::Integer)
    size = length(chain.algorithms)
    @assert size > 0

    result = fit(chain.algorithms[1], data, k)
    for i in 2:size
        result = fit(chain.algorithms[i], data, result)
    end
    return result
end

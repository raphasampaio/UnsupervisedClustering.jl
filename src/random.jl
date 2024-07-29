function sample_unique_data(rng::AbstractRNG, data::AbstractMatrix{<:Real}, k::Integer)
    unique_data = unique(data, dims = 1)
    unique_size = size(unique_data, 1)
    indices = StatsBase.sample(rng, 1:unique_size, k, replace = unique_size < k)
    return unique_data, indices
end

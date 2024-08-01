function try_sampling_unique_data(rng::AbstractRNG, data::AbstractMatrix{<:Real}, k::Integer)
    unique_data = unique(data, dims = 1)
    unique_size = size(unique_data, 1)

    if unique_size < k
        n, d = size(data)
        indices = StatsBase.sample(rng, 1:n, k, replace = false)
        return data, indices
    else
        indices = StatsBase.sample(rng, 1:unique_size, k, replace = false)
        return unique_data, indices
    end
end

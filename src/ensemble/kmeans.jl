function fit(algorithm::Kmeans, data::AbstractMatrix{<:Real}, result_gmm::GMMResult)
    result_kmeans = convert(Kmeans, result_gmm)
    fit!(algorithm, data, result_kmeans)
    return result_kmeans
end
function fit(algorithm::GMM, data::AbstractMatrix{<:Real}, result_kmeans::KmeansResult)
    result_gmm = convert(GMMResult, result_kmeans)
    fit!(algorithm, data, result_gmm)
    return result_gmm
end
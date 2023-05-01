function EnsembleClustering(algorithms::ClusteringAlgorithm...)
    return EnsembleClustering(collect(algorithms))
end

function fit(ensemble::EnsembleClustering, data::AbstractMatrix{<:Real}, k::Integer)
    size = length(ensemble.algorithms)
    @assert size > 0

    result = fit(ensemble.algorithms[1], data, k)
    for i in 2:size
        result = fit(ensemble.algorithms[i], data, result)
    end
    return result
end

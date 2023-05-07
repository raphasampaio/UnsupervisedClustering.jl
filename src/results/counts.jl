function counts(result::ClusteringResult)
    return StatsBase.counts(result.assignments, result.k)
end

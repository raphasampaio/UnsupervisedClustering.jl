function counts(result::UnsupervisedClusteringResult)
    return StatsBase.counts(result.assignments, result.k)
end

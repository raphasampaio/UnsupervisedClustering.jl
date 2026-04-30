function counts(result::AbstractResult)
    return StatsBase.counts(result.assignments, result.k)
end

function has_empty_clusters(result::AbstractResult)
    return any(==(0), counts(result))
end
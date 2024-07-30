function counts(result::AbstractResult)
    return StatsBase.counts(result.assignments, result.k)
end

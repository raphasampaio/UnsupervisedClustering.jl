function counts(result::Result)
    return StatsBase.counts(result.assignments, result.k)
end

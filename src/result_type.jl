function result_type(::Type{GMM})
    return GMMResult
end

function result_type(::Type{Kmeans})
    return KmeansResult
end

function result_type(::Type{BalancedKmeans})
    return KmeansResult
end

function result_type(::Type{Kmedoids})
    return KmedoidsResult
end

function result_type(::Type{BalancedKmedoids})
    return KmedoidsResult
end

function result_type(::Type{Ksegmentation})
    return KsegmentationResult
end
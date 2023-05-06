function isbetter(a::KmeansResult, b::KmeansResult)
    return isless(a.objective, b.objective)
end

function isbetter(a::KmedoidsResult, b::KmedoidsResult)
    return isless(a.objective, b.objective)
end

function isbetter(a::GMMResult, b::GMMResult)
    return isless(b.objective, a.objective)
end

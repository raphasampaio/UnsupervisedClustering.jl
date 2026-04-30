module TestDiscardEmptyClusters

using Random
using Test
using UnsupervisedClustering

Base.@kwdef struct EmptyingLocalSearch <: UnsupervisedClustering.AbstractAlgorithm
    rng::AbstractRNG = MersenneTwister(0)
end

function UnsupervisedClustering.fit(::EmptyingLocalSearch, data::AbstractMatrix{<:Real}, k::Integer)
    n, d = size(data)
    result = UnsupervisedClustering.KmeansResult(zeros(Int, n), zeros(d, k))
    for i in 1:n
        result.assignments[i] = mod(i - 1, k) + 1
    end
    result.objective = 100.0
    return result
end

function UnsupervisedClustering.fit!(
    ::EmptyingLocalSearch,
    ::AbstractMatrix{<:Real},
    result::UnsupervisedClustering.KmeansResult,
)
    fill!(result.assignments, 1)
    result.objective = 1.0
    return nothing
end

@testset "RandomSwap - discard_empty_clusters" begin
    data = rand(MersenneTwister(1), 50, 2)
    k = 4

    no_discard = RandomSwap(
        local_search = EmptyingLocalSearch(rng = MersenneTwister(2)),
        max_iterations = 5,
        max_iterations_without_improvement = 100,
        discard_empty_clusters = false,
        verbose = true,
    )
    result_no_discard = UnsupervisedClustering.fit(no_discard, data, k)
    @test count(==(0), UnsupervisedClustering.counts(result_no_discard)) == k - 1
    @test result_no_discard.objective == 1.0

    discard = RandomSwap(
        local_search = EmptyingLocalSearch(rng = MersenneTwister(2)),
        max_iterations = 5,
        max_iterations_without_improvement = 100,
        discard_empty_clusters = true,
        verbose = true,
    )
    result_discard = UnsupervisedClustering.fit(discard, data, k)
    @test count(==(0), UnsupervisedClustering.counts(result_discard)) == 0
    @test result_discard.objective == 100.0

    return nothing
end

end

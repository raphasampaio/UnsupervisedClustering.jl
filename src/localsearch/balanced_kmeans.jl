function BalancedKmeans(;
    metric::SemiMetric = SqEuclidean()
    verbose::Bool = DEFAULT_VERBOSE
    rng::AbstractRNG = Random.GLOBAL_RNG
    tolerance::Real = DEFAULT_TOLERANCE
    max_iterations::Integer = DEFAULT_MAX_ITERATIONS
)
    return Kmeans(
        metric=metric,
        verbose=verbose,
        rng=rng,
        tolerance=tolerance,
        max_iterations=max_iterations,
        assignment_step=balanced_kmeans_assignment_step!,
    )
end

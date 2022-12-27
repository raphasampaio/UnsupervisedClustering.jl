function run_algorithm(
    label::String, 
    algorithm::UnsupervisedClustering.Algorithm, 
    data::AbstractMatrix{<:Real}, 
    k::Int, 
    expected::AbstractVector{<:Integer},
)
    @timeit label result = UnsupervisedClustering.train(algorithm, data, k)
    objective = result.objective
    assignments = result.assignments
    ari = Clustering.randindex(assignments, expected)[1]

    println("$label - $objective - $ari")
    return objective, assignments, ari
end

function test_gmm(data::AbstractMatrix{<:Real}, k::Int, expected::AbstractVector{<:Integer}, seed::Integer)
    n, d = size(data)

    empirical = EmpiricalCovarianceMatrix(n, d)
    shrunk = ShrunkCovarianceMatrix(n, d)
    oas = OASCovarianceMatrix(n, d)
    lw = LedoitWolfCovarianceMatrix(n, d)

    gmm_sk = GMMSK(rng = MersenneTwister(seed))

    gmm = GMM(estimator = empirical, rng = MersenneTwister(seed))
    gmm_ms = MultiStart(local_search = gmm)
    gmm_rs = RandomSwap(local_search = gmm)
    gmm_hg = GeneticAlgorithm(local_search = gmm)

    gmm_shrunk = GMM(estimator = shrunk, rng = MersenneTwister(seed))
    gmm_shrunk_ms = MultiStart(local_search = gmm_shrunk)
    gmm_shrunk_rs = RandomSwap(local_search = gmm_shrunk)
    gmm_shrunk_hg = GeneticAlgorithm(local_search = gmm_shrunk)

    gmm_oas = GMM(estimator = oas, rng = MersenneTwister(seed))
    gmm_oas_ms = MultiStart(local_search = gmm_oas)
    gmm_oas_rs = RandomSwap(local_search = gmm_oas)
    gmm_oas_hg = GeneticAlgorithm(local_search = gmm_oas)

    gmm_lw = GMM(estimator = lw, rng = MersenneTwister(seed))
    gmm_lw_ms = MultiStart(local_search = gmm_lw)
    gmm_lw_rs = RandomSwap(local_search = gmm_lw)
    gmm_lw_hg = GeneticAlgorithm(local_search = gmm_lw)

    objective_sk, assignments_sk, ari_sk = run_algorithm("gmm_sk       ", gmm_sk, data, k, expected)
    objective_ss, assignments_ss, ari_ss = run_algorithm("gmm          ", gmm, data, k, expected)
    objective_ms, assignments_ms, ari_ms = run_algorithm("gmm_ms       ", gmm_ms, data, k, expected)
    objective_rs, assignments_rs, ari_rs = run_algorithm("gmm_rs       ", gmm_rs, data, k, expected)
    objective_hg, assignments_hg, ari_hg = run_algorithm("gmm_hg       ", gmm_hg, data, k, expected)

    objective_shrunk_ss, assignments_shrunk_ss, ari_shrunk_ss  = run_algorithm("gmm_shrunk   ", gmm_shrunk, data, k, expected)
    objective_shrunk_ms, assignments_shrunk_ms, ari_shrunk_ms  = run_algorithm("gmm_shrunk_ms", gmm_shrunk_ms, data, k, expected)
    objective_shrunk_rs, assignments_shrunk_rs, ari_shrunk_rs  = run_algorithm("gmm_shrunk_rs", gmm_shrunk_rs, data, k, expected)
    objective_shrunk_hg, assignments_shrunk_hg, ari_shrunk_hg  = run_algorithm("gmm_shrunk_hg", gmm_shrunk_hg, data, k, expected)

    objective_oas_ss, assignments_oas_ss, ari_oas_ss  = run_algorithm("gmm_oas   ", gmm_oas, data, k, expected)
    objective_oas_ms, assignments_oas_ms, ari_oas_ms  = run_algorithm("gmm_oas_ms", gmm_oas_ms, data, k, expected)
    objective_oas_rs, assignments_oas_rs, ari_oas_rs  = run_algorithm("gmm_oas_rs", gmm_oas_rs, data, k, expected)
    objective_oas_hg, assignments_oas_hg, ari_oas_hg  = run_algorithm("gmm_oas_hg", gmm_oas_hg, data, k, expected)

    objective_lw_ss, assignments_lw_ss, ari_lw_ss  = run_algorithm("gmm_lw   ", gmm_lw, data, k, expected)
    objective_lw_ms, assignments_lw_ms, ari_lw_ms  = run_algorithm("gmm_lw_ms", gmm_lw_ms, data, k, expected)
    objective_lw_rs, assignments_lw_rs, ari_lw_rs  = run_algorithm("gmm_lw_rs", gmm_lw_rs, data, k, expected)
    objective_lw_hg, assignments_lw_hg, ari_lw_hg  = run_algorithm("gmm_lw_hg", gmm_lw_hg, data, k, expected)

    @test ari_ss ≈ ari_sk
    # @test objective_ss <= objective_ms
    # @test objective_ss <= objective_rs
    # @test objective_ss <= objective_hg

    return
end
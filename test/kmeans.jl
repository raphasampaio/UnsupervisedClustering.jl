function test_kmeans(data::AbstractMatrix{<:Real}, k::Int, expected::AbstractVector{<:Integer})
    d, n = size(data)
    Random.seed!(1)
    permutation = randperm(n)

    # KMEANS - SINGLE START
    result_ss = UnsupervisedClustering.KmeansResult(d, n, k)
    for i in 1:d
        for j in 1:k
            result_ss.centers[i, j] = data[i, permutation[j]]
        end
    end
    Random.seed!(1)
    kmeans = Kmeans(
        verbose = false, 
        metric = SqEuclidean(), 
        tolerance = 1e-3, 
        max_iterations = 1000
    )
    @timeit "kmeans - ss" UnsupervisedClustering.train!(kmeans, data, result_ss)
    objective_ss = result_ss.objective
    assignments_ss = result_ss.assignments

    # KMEANS - CLUSTERING.JL
    Random.seed!(1)
    @timeit "kmeans - jl" result_jl = Clustering.kmeans(
        data,
        k,
        init = permutation[1:k],
        display = :none,
        distance = SqEuclidean(),
        tol = 1e-3,
        maxiter = 1000,
    )
    objective_jl = result_jl.totalcost
    assignments_jl = result_jl.assignments

    # KMEANS - SKLEARN
    centers_sk = zeros(k, d)
    for i in 1:k
        for j in 1:d
            centers_sk[i, j] = data[i, permutation[j]]
        end
    end
    Random.seed!(1)
    kmeans_sk = KMeans(
        n_clusters = k,
        init = centers_sk,
        n_init = 1,
        max_iter = 1000,
        tol = 1e-6,
        verbose = 0,
        # random_state=None, 
        # copy_x=true, 
        algorithm = "lloyd"
    )
    @timeit "kmeans - sk" assignments_sk = fit_predict!(kmeans_sk, data')
    objective_sk = -score(kmeans_sk, data')

    # KMEANS - MULTI-START
    Random.seed!(1)
    multistart = MultiStart(local_search = kmeans)
    @timeit "kmeans - ms" result_ms = UnsupervisedClustering.train(multistart, data, k)
    objective_ms = result_ms.objective
    assignments_ms = result_ms.assignments

    # KMEANS - RANDOM SWAP
    Random.seed!(1)
    randomswap = RandomSwap(local_search = kmeans)
    @timeit "kmeans - rs" result_rs = UnsupervisedClustering.train(randomswap, data, k)
    objective_rs = result_rs.objective
    assignments_rs = result_rs.assignments

    # KMEANS - GENETIC-ALGORITHM
    Random.seed!(1)
    genetic = GeneticAlgorithm(local_search = kmeans)
    @timeit "kmeans - hg" result_hg = UnsupervisedClustering.train(genetic, data, k)
    objective_hg = result_hg.objective
    assignments_hg = result_hg.assignments

    ari_ss = Clustering.randindex(assignments_ss, expected)[1]
    ari_jl = Clustering.randindex(assignments_jl, expected)[1]
    ari_sk = Clustering.randindex(assignments_sk, expected)[1]
    ari_ms = Clustering.randindex(assignments_ms, expected)[1]
    ari_rs = Clustering.randindex(assignments_rs, expected)[1]
    ari_hg = Clustering.randindex(assignments_hg, expected)[1]

    println("$objective_ss - $ari_ss")
    println("$objective_jl - $ari_jl")
    println("$objective_sk - $ari_sk")
    println("$objective_ms - $ari_ms")
    println("$objective_rs - $ari_rs")
    println("$objective_hg - $ari_hg")

    @test ari_ss ≈ ari_jl
    @test objective_ss >= objective_ms

    return
end

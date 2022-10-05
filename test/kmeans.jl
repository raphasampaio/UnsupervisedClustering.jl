function test_kmeans(data::Matrix{Float64}, k::Integer, expected::Vector{Int}, seed::Integer = 1, verbose::Bool = false)
    n = size(data, 1)
    d = size(data, 2)

    Random.seed!(seed)
    permutation = randperm(n)

    centers = zeros(k, d)
    for i in 1:k
        for j in 1:d
            centers[i, j] = data[permutation[i], j]
        end
    end

    Random.seed!(seed)
    assignments1 = UnsupervisedClustering.kmeans(data, k).assignments

    Random.seed!(seed)
    model2 = KMeans(
        n_clusters = k,
        init = centers,
        n_init = 1,
        max_iter = 10000,
        tol = 1e-4,
        verbose = verbose ? 1 : 0,
        # random_state=None, 
        # copy_x=true, 
        algorithm = "lloyd"
    )
    assignments2 = fit_predict!(model2, data)

    Random.seed!(seed)
    model3 = Clustering.kmeans(data', k, init = permutation[1:k], display = verbose ? :iter : :none, tol = 1e-3, maxiter = 10000)
    assignments3 = model3.assignments

    @show ari1 = Clustering.randindex(assignments1, expected)[1]
    @show ari2 = Clustering.randindex(assignments2, expected)[1]
    @show ari3 = Clustering.randindex(assignments3, expected)[1]

    @test ari1 ≈ ari2
    @test ari1 ≈ ari3

    return nothing
end

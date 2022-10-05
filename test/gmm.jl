function test_gmm(data::Matrix{Float64}, k::Integer, expected::Vector{Int}, seed::Integer = 1, verbose::Bool = false)
    n = size(data, 1)
    d = size(data, 2)

    Random.seed!(seed)
    permutation = randperm(n)

    μ = zeros(k, d)
    for i in 1:k
        for j in 1:d
            μ[i, j] = data[permutation[i], j]
        end
    end

    Σ1 = zeros(k, d, d)
    for i in 1:k
        for j in 1:d
            Σ1[i, j, j] = 1.0
        end
    end

    Σ2 = Vector{UpperTriangular{Float64,Matrix{Float64}}}()
    for i in 1:k
        push!(Σ2, cholesky(Matrix{Float64}(I, d, d)).U)
    end

    weights = ones(k) ./ k

    Random.seed!(seed)
    assignments1 = UnsupervisedClustering.gmm(data, k).assignments

    Random.seed!(seed)
    model2 = GaussianMixture(
        n_components = k,
        covariance_type = "full",
        tol = 1e-3,
        reg_covar = 0.0,
        max_iter = 10000,
        n_init = 1,
        weights_init = weights,
        means_init = μ,
        precisions_init = Σ1,
        verbose = verbose ? 1 : 0,
        verbose_interval = 1
    )
    assignments2 = fit_predict!(model2, data)

    Random.seed!(seed)
    history = Vector{History}()
    model3 = GMM(weights, μ, Σ2, history, 0)
    em!(model3, data, nIter = 40, varfloor = 1e-3)
    likelihood = llpg(model3, data)
    assignments3 = zeros(Int, n)
    for i in 1:n
        assignments3[i] = argmax(likelihood[i, :])
    end

    @show ari1 = Clustering.randindex(assignments1, expected)[1]
    @show ari2 = Clustering.randindex(assignments2, expected)[1]
    @show ari3 = Clustering.randindex(assignments3, expected)[1]

    @test ari1 ≈ ari2

    return nothing
end

using UnsupervisedClustering

using Aqua
using DelimitedFiles
using Distances
using LinearAlgebra
using Printf
using Random
using RegularizedCovarianceMatrices
using StableRNGs
using Test
using TimerOutputs

function get_data(filename::String)
    open(joinpath("data", "$filename.csv")) do file
        table = readdlm(file, ',')
        n = size(table, 1)

        clusters = Set{Int}()
        for i in 1:n
            expected = Int(table[i, 1])
            push!(clusters, expected)
        end
        k = length(clusters)

        return table[:, 2:size(table, 2)], k
    end
end

function test_aqua()
    @testset "Ambiguities" begin
        Aqua.test_ambiguities(UnsupervisedClustering, recursive = false)
    end
    Aqua.test_all(UnsupervisedClustering, ambiguities = false)
    return nothing
end

function test_all()
    # println("BLAS: $(BLAS.get_config())")

    @testset "Aqua.jl" begin
        test_aqua()
    end

    @testset "n = 0" begin
        n, d, k = 0, 2, 3
        data = zeros(n, d)

        @testset "kmeans" begin
            algorithm = Kmeans(rng = StableRNG(1))
            result = fit(algorithm, data, k)
            @test length(result.assignments) == 0

            result = fit(algorithm, data, Vector{Int}())
            @test length(result.assignments) == 0
        end

        @testset "balanced kmeans" begin
            algorithm = BalancedKmeans(rng = StableRNG(1))
            result = fit(algorithm, data, k)
            @test length(result.assignments) == 0

            result = fit(algorithm, data, Vector{Int}())
            @test length(result.assignments) == 0
        end

        @testset "ksegmentation" begin
            algorithm = Ksegmentation()
            result = fit(algorithm, data, k)
            @test length(result.assignments) == 0
        end

        @testset "kmedoids" begin
            algorithm = Kmedoids(rng = StableRNG(1))
            distances = pairwise(SqEuclidean(), data, dims = 1)
            result = fit(algorithm, distances, k)
            @test length(result.assignments) == 0

            result = fit(algorithm, data, Vector{Int}())
            @test length(result.assignments) == 0
        end

        @testset "gmm" begin
            algorithm = GMM(rng = StableRNG(1), estimator = EmpiricalCovarianceMatrix(n, d))
            result = fit(algorithm, data, k)
            @test length(result.assignments) == 0

            result = fit(algorithm, data, Vector{Int}())
            @test length(result.assignments) == 0
        end
    end

    @testset "d = 0" begin
        n, d, k = 3, 0, 3
        data = zeros(n, d)

        @testset "kmeans" begin
            algorithm = Kmeans(rng = StableRNG(1))
            @test_throws AssertionError result = fit(algorithm, data, k)
        end

        @testset "balanced kmeans" begin
            algorithm = BalancedKmeans(rng = StableRNG(1))
            @test_throws AssertionError result = fit(algorithm, data, k)
        end

        @testset "ksegmentation" begin
            algorithm = Ksegmentation()
            @test_throws AssertionError result = fit(algorithm, data, k)
        end

        @testset "gmm" begin
            algorithm = GMM(rng = StableRNG(1), estimator = EmpiricalCovarianceMatrix(n, d))
            @test_throws AssertionError result = fit(algorithm, data, k)
        end
    end

    @testset "k > n" begin
        n, d, k = 2, 2, 3
        data = zeros(n, d)

        @testset "kmeans" begin
            algorithm = Kmeans(rng = StableRNG(1))
            @test_throws AssertionError result = fit(algorithm, data, k)
        end

        @testset "balanced kmeans" begin
            algorithm = BalancedKmeans(rng = StableRNG(1))
            @test_throws AssertionError result = fit(algorithm, data, k)
        end

        @testset "ksegmentation" begin
            algorithm = Ksegmentation()
            @test_throws AssertionError result = fit(algorithm, data, k)
        end

        @testset "kmedoids" begin
            algorithm = Kmedoids(rng = StableRNG(1))
            distances = pairwise(SqEuclidean(), data, dims = 1)
            @test_throws AssertionError result = fit(algorithm, distances, k)
        end

        @testset "gmm" begin
            algorithm = GMM(rng = StableRNG(1), estimator = EmpiricalCovarianceMatrix(n, d))
            @test_throws AssertionError result = fit(algorithm, data, k)
        end
    end

    @testset "n = k" begin
        n, d, k = 3, 2, 3
        data = rand(StableRNG(1), n, d)

        @testset "kmeans" begin
            algorithm = Kmeans(rng = StableRNG(1))
            result = fit(algorithm, data, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end

        @testset "balanced kmeans" begin
            algorithm = BalancedKmeans(rng = StableRNG(1))
            result = fit(algorithm, data, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end

        @testset "ksegmentation" begin
            algorithm = Ksegmentation()
            result = fit(algorithm, data, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end

        @testset "kmedoids" begin
            algorithm = Kmedoids(rng = StableRNG(1))
            distances = pairwise(SqEuclidean(), data, dims = 1)
            result = fit(algorithm, distances, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end

        @testset "gmm" begin
            algorithm = GMM(rng = StableRNG(1), estimator = EmpiricalCovarianceMatrix(n, d))
            result = fit(algorithm, data, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end
    end

    @testset "n = k (zeros)" begin
        n, d, k = 3, 2, 3
        data = zeros(n, d)

        @testset "kmeans" begin
            algorithm = Kmeans(rng = StableRNG(1))
            result = fit(algorithm, data, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end

        @testset "balanced kmeans" begin
            algorithm = BalancedKmeans(rng = StableRNG(1))
            result = fit(algorithm, data, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end

        @testset "kmedoids" begin
            algorithm = Kmedoids(rng = StableRNG(1))
            distances = pairwise(SqEuclidean(), data, dims = 1)
            result = fit(algorithm, distances, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end

        @testset "gmm" begin
            algorithm = GMM(rng = StableRNG(1), estimator = EmpiricalCovarianceMatrix(n, d))
            result = fit(algorithm, data, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end
    end

    @testset "convert" begin
        kmeans_result = UnsupervisedClustering.KmeansResult([1, 2, 2], [1.0 2.0; 1.0 2.0; 1.0 2.0])
        gmm_result = UnsupervisedClustering.GMMResult(
            [1, 2, 2],
            [0.5, 0.5],
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [Symmetric(Matrix{Float64}(I, 3, 3)) for _ in 1:2],
        )

        @test kmeans_result == convert(UnsupervisedClustering.KmeansResult, gmm_result)
        @test gmm_result == convert(UnsupervisedClustering.GMMResult, kmeans_result)
    end

    @testset "concatenate" begin
        @test_throws MethodError concatenate()

        @testset "kmeans result" begin
            result = concatenate(
                UnsupervisedClustering.KmeansResult([1, 2], [1.0 2.0; 1.0 2.0; 1.0 2.0], 1.0, [0.5, 0.5], 1, 1.0, true),
                UnsupervisedClustering.KmeansResult([1, 2], [1.0 2.0; 1.0 2.0; 1.0 2.0], 2.0, [1.0, 1.0], 2, 2.0, true),
                UnsupervisedClustering.KmeansResult([1, 2, 2], [1.0 2.0; 1.0 2.0; 1.0 2.0], 3.0, [1.5, 1.5], 3, 3.0, true),
            )

            @test result.k == 6
            @test result.assignments == [1, 2, 3, 4, 5, 6, 6]
            @test result.clusters ≈ [1.0 2.0 1.0 2.0 1.0 2.0; 1.0 2.0 1.0 2.0 1.0 2.0; 1.0 2.0 1.0 2.0 1.0 2.0]
            @test result.objective ≈ 6.0
            @test result.objective_per_cluster ≈ [0.5, 0.5, 1.0, 1.0, 1.5, 1.5]
            @test result.iterations == 6
            @test result.elapsed ≈ 6.0
            @test result.converged == true
        end

        @testset "ksegmentation result" begin
            result = concatenate(
                UnsupervisedClustering.KsegmentationResult([1, 2], [1.0 2.0; 1.0 2.0; 1.0 2.0], 1.0, [0.5, 0.5], 1, 1.0, true),
                UnsupervisedClustering.KsegmentationResult([1, 2], [1.0 2.0; 1.0 2.0; 1.0 2.0], 2.0, [1.0, 1.0], 2, 2.0, true),
                UnsupervisedClustering.KsegmentationResult([1, 2, 2], [1.0 2.0; 1.0 2.0; 1.0 2.0], 3.0, [1.5, 1.5], 3, 3.0, true),
            )

            @test result.k == 6
            @test result.assignments == [1, 2, 3, 4, 5, 6, 6]
            @test result.clusters ≈ [1.0 2.0 1.0 2.0 1.0 2.0; 1.0 2.0 1.0 2.0 1.0 2.0; 1.0 2.0 1.0 2.0 1.0 2.0]
            @test result.objective ≈ 6.0
            @test result.objective_per_cluster ≈ [0.5, 0.5, 1.0, 1.0, 1.5, 1.5]
            @test result.iterations == 6
            @test result.elapsed ≈ 6.0
            @test result.converged == true
        end

        @testset "kmedoids result" begin
            result = concatenate(
                UnsupervisedClustering.KmedoidsResult([1, 2], [1, 2], 1.0, [0.5, 0.5], 1, 1.0, true),
                UnsupervisedClustering.KmedoidsResult([1, 2], [1, 2], 2.0, [1.0, 1.0], 2, 2.0, true),
                UnsupervisedClustering.KmedoidsResult([1, 2, 2], [1, 2], 3.0, [1.5, 1.5], 3, 3.0, true),
            )

            @test result.k == 6
            @test result.assignments ≈ [1, 2, 3, 4, 5, 6, 6]
            @test result.clusters == [1, 2, 3, 4, 5, 6]
            @test result.objective ≈ 6.0
            @test result.objective_per_cluster ≈ [0.5, 0.5, 1.0, 1.0, 1.5, 1.5]
            @test result.iterations == 6
            @test result.elapsed ≈ 6.0
            @test result.converged == true
        end
    end

    @testset "sort" begin
        @testset "kmeans result" begin
            result = UnsupervisedClustering.KmeansResult([1, 2, 3, 3, 2, 1], [3.0 1.0 2.0; 3.0 1.0 2.0], 6.0, [3.0, 1.0, 2.0], 1, 1.0, true)
            sort!(result)

            @test result.k == 3
            @test result.assignments == [3, 1, 2, 2, 1, 3]
            @test result.clusters ≈ [1.0 2.0 3.0; 1.0 2.0 3.0]
            @test result.objective ≈ 6.0
            @test result.objective_per_cluster ≈ [1.0, 2.0, 3.0]
            @test result.iterations == 1
            @test result.elapsed ≈ 1.0
            @test result.converged == true
        end

        @testset "kmedoids result" begin
            result = UnsupervisedClustering.KmedoidsResult([1, 2, 3, 3, 2, 1], [3, 1, 2], 6.0, [3.0, 1.0, 2.0], 1, 1.0, true)
            sort!(result)

            @test result.k == 3
            @test result.assignments == [3, 1, 2, 2, 1, 3]
            @test result.clusters ≈ [1, 2, 3]
            @test result.objective ≈ 6.0
            @test result.objective_per_cluster ≈ [1.0, 2.0, 3.0]
            @test result.iterations == 1
            @test result.elapsed ≈ 1.0
            @test result.converged == true
        end
    end

    @testset "markov" begin
        result1 = UnsupervisedClustering.KmeansResult([1, 2, 3, 3, 2, 1], zeros(0, 3), 0.0, zeros(0), 0, 0.0, false)
        result2 = UnsupervisedClustering.KmeansResult([2, 3, 1, 3, 1, 2], zeros(0, 3), 0.0, zeros(0), 0, 0.0, false)

        @test stochastic_matrix(result1, result2) ≈ [0.0 1.0 0.0; 0.5 0.0 0.5; 0.5 0.0 0.5]
    end

    verbose = true
    decompose_if_fails = true
    max_iterations = 30
    max_iterations_without_improvement = 15

    datasets = Dict(
        #! format: off
        "3_2_-0.26" => [26416.7106071853449976,26399.3161993783432990,26399.3161993783432990,26399.3161993783432990,36147.5081656920665409,36147.5081656920665409,36073.5957067493145587,36073.5957067493145587,9345113.0105678215622902,26750.1476956724618503,26654.1405923047605029,26654.1405923047605029,26654.1405923047605029,-7.4702212739335767,-7.4500211230970370,-7.4489366154226495,-7.4500211230970370,-7.4725939132511821,-7.4526387237766327,-7.4533339416188955,-7.4521420085002443,-7.4722211994025169,-7.4510206920328903,-7.4493953372850443,-7.4497226376357908,-7.4722211994025187,-7.4510206920328903,-7.4493953372850426,-7.4497226376357881,-7.4635411676483958,26416.7106071854177571,],
        "3_2_-0.1" => [59601.5900345181798912,59567.5943560912564863,59567.5943560912564863,59567.5943560912564863,63170.3206470118893776,63170.3206470118893776,63170.3206470118893776,63170.3206470118893776,18987340.7112460732460022,60614.1308034227404278,60439.0608680820660084,60439.0608680820660084,60439.0608680820660084,-8.4651467479894027,-8.4102349541014298,-8.4100334768429441,-8.4101824043248463,-8.4699664124117344,-8.4115802534317705,-8.4115408216358656,-8.4115802534317705,-8.4681605617681157,-8.4107629148135796,-8.4106804571296419,-8.4105989765192977,-8.4681605617681193,-8.4107629148135814,-8.4106804571296347,-8.4105989765192977,-8.4110546385982605,59601.5900345181871671,],
        "3_2_0.01" => [75954.5837682041164953,75940.7629553540900815,75940.7629553540900815,75940.7629553540900815,89184.4377358829224249,89184.4377358829224249,89184.4377358829224249,89184.4377358829224249,19017664.5753948241472244,77146.2837541921180673,77146.2837541921180673,77146.2837541921180673,77146.2837541921180673,-8.5182026740741890,-8.5180572297950512,-8.5181243009978562,-8.5180572297950512,-8.5319332437570576,-8.5285204304284701,-8.5292615394819506,-8.5285204304284701,-8.5244412706655197,-8.5215019005106267,-8.5226478094182152,-8.5215019005106267,-8.5244412706655162,-8.5215019005106214,-8.5226478094182117,-8.5215019005106196,-8.5181717001935695,75954.5837682037526974,],
        "3_2_0.21" => [54050.7068362436621101,54050.7068362436621101,54050.7068362436621101,54050.7068362436621101,58109.7044917582825292,58109.7044917582825292,58109.7044917582825292,58109.7044917582825292,28577467.4943606853485107,190748.1609571795270313,54294.3046370887896046,54294.3046370887896046,54294.3046370887896046,-8.2263103563940216,-8.2262481992798620,-8.2262482876317655,-8.2262481992798620,-8.6237533552707273,-8.2299991802277717,-8.2301746066156909,-8.2299632663079993,-8.2270765382688573,-8.2267451448259443,-8.2267842529025579,-8.2267351688995429,-8.2270765382688538,-8.2267451448259408,-8.2267842529025579,-8.2267351688995429,-8.2263106409859788,54050.7068362438221811,],
        "3_5_-0.26" => [145171.0541117230604868,145171.0541117230604868,145171.0541117230604868,145171.0541117230604868,150284.6343761875468772,146995.5582167978864163,147717.0315828337043058,146995.5582167978864163,16613451.2244664989411831,164210.6193630891793873,156312.0316052992711775,157617.3088592314161360,156312.0316052992711775,-18.9902279628581034,-18.9894390359008867,-18.9886866180432072,-18.9889584644526543,-19.0647732702474251,-19.0568682965651384,-19.0529243169820610,-19.0524957912762467,-19.0110018656908721,-19.0084532025838726,-19.0089899499282780,-19.0069350277853388,-19.0110018656908686,-19.0084532025838797,-19.0089899499282744,-19.0069350277853459,-19.0004688481004393,145189.9506311327277217,],
        "3_5_-0.1" => [140639.1790945607062895,140633.0693809580116067,140633.0693809580116067,140633.0693809580116067,151226.4580142682825681,147724.2177938571549021,148850.7255250816233456,147724.2177938571549021,28741592.7844186127185822,167245.8189150553662330,146641.2836113125958946,146811.0744057828269433,146641.2836113125958946,-18.5691678679363257,-18.4751111674275457,-18.5667407402900153,-18.4750896652624839,-18.7486347466308345,-18.7370213294714674,-18.7377183979080577,-18.7324367890290411,-18.6485306675739366,-18.6384134032780686,-18.6337407395391992,-18.6365737875567596,-18.6485306675739366,-18.6384134032780686,-18.6337407395392027,-18.6365737875567667,-18.5707220843082830,140658.3377095228352118,],
        "3_5_0.01" => [186156.1504602948843967,186118.1492456819978543,186118.1492456819978543,186118.1492456819978543,205511.4498680337273981,204413.6498413142398931,204413.6498413142398931,204413.6498413142398931,29641666.3993386998772621,196657.4337840474909171,196389.1341025999281555,196389.1341025999281555,196389.1341025999281555,-19.8157275913082671,-19.8151937057868253,-19.8150151774982923,-19.8151129297369870,-19.8308307696539465,-19.8306611769480803,-19.8308307696539465,-19.8306611769480803,-19.8163141943443151,-19.8163141943443151,-19.8160679672704489,-19.8160072457278140,-19.8163141943443151,-19.8163141943443151,-19.8160679672704418,-19.8160072457278176,-19.8152166973165720,186156.1504602951463312,],
        "3_5_0.21" => [115218.3593000062537612,115205.1984080591937527,115205.1984080591937527,115205.1984080591937527,121752.4784170263010310,121752.4784170263010310,121752.4784170263010310,121752.4784170263010310,19819490.7155972793698311,120734.2042795864981599,120734.2042795864981599,120734.2042795864981599,120734.2042795864981599,-18.3813267993559428,-18.3813152644100271,-18.3813163090532719,-18.3813150326683896,-18.3997368285819967,-18.3996036784888339,-18.3996001823870756,-18.3995578336385321,-18.3819556412348639,-18.3819335859311934,-18.3819335532215611,-18.3819335859311899,-18.3819556412348533,-18.3819335859311863,-18.3819335532215575,-18.3819335859311934,-18.3814368484425010,115218.3593000061518978,],
        #! format: on
    )

    for (dataset, benchmark) in datasets
        data, k = get_data(dataset)
        n, d = size(data)

        kmeans = Kmeans(
            verbose = verbose,
            rng = StableRNG(1),
        )

        balanced_kmeans = BalancedKmeans(
            verbose = verbose,
            rng = StableRNG(1),
        )

        ksegmentation = Ksegmentation()

        kmedoids = Kmedoids(
            verbose = verbose,
            rng = StableRNG(1),
        )

        gmm = GMM(
            verbose = verbose,
            rng = StableRNG(1),
            estimator = EmpiricalCovarianceMatrix(n, d),
            decompose_if_fails = decompose_if_fails,
        )

        gmm_shrunk = GMM(
            verbose = verbose,
            rng = StableRNG(1),
            estimator = ShrunkCovarianceMatrix(n, d),
            decompose_if_fails = decompose_if_fails,
        )

        gmm_oas = GMM(
            verbose = verbose,
            rng = StableRNG(1),
            estimator = LedoitWolfCovarianceMatrix(n, d),
            decompose_if_fails = decompose_if_fails,
        )

        gmm_lw = GMM(
            verbose = verbose,
            rng = StableRNG(1),
            estimator = LedoitWolfCovarianceMatrix(n, d),
            decompose_if_fails = decompose_if_fails,
        )

        algorithms = [
            # KMEANS
            kmeans,
            MultiStart(
                local_search = kmeans,
                verbose = verbose,
                max_iterations = max_iterations,
            ),
            RandomSwap(
                local_search = kmeans,
                verbose = verbose,
                max_iterations = max_iterations,
                max_iterations_without_improvement = max_iterations_without_improvement,
            ),
            GeneticAlgorithm(
                local_search = kmeans,
                verbose = verbose,
                max_iterations = max_iterations,
                max_iterations_without_improvement = max_iterations_without_improvement,
            ),
            # BALANCED KMEANS
            balanced_kmeans,
            MultiStart(
                local_search = balanced_kmeans,
                verbose = verbose,
                max_iterations = max_iterations,
            ),
            RandomSwap(
                local_search = balanced_kmeans,
                verbose = verbose,
                max_iterations = max_iterations,
                max_iterations_without_improvement = max_iterations_without_improvement,
            ),
            GeneticAlgorithm(
                local_search = balanced_kmeans,
                verbose = verbose,
                max_iterations = max_iterations,
                max_iterations_without_improvement = max_iterations_without_improvement,
            ),
            # KSEGMENTATION
            ksegmentation,
            # KMEDOIDS
            kmedoids,
            MultiStart(
                local_search = kmedoids,
                verbose = verbose,
                max_iterations = max_iterations,
            ),
            RandomSwap(
                local_search = kmedoids,
                verbose = verbose,
                max_iterations = max_iterations,
                max_iterations_without_improvement = max_iterations_without_improvement,
            ),
            GeneticAlgorithm(
                local_search = kmedoids,
                verbose = verbose,
                max_iterations = max_iterations,
                max_iterations_without_improvement = max_iterations_without_improvement,
            ),
            # GMM EMPIRICAL
            gmm,
            MultiStart(
                local_search = gmm,
                max_iterations = max_iterations,
            ),
            RandomSwap(
                local_search = gmm,
                verbose = verbose,
                max_iterations = max_iterations,
                max_iterations_without_improvement = max_iterations_without_improvement,
            ),
            GeneticAlgorithm(
                local_search = gmm,
                verbose = verbose,
                max_iterations = max_iterations,
                max_iterations_without_improvement = max_iterations_without_improvement,
            ),
            # GMM SHRUNK
            gmm_shrunk,
            MultiStart(
                local_search = gmm_shrunk,
                max_iterations = max_iterations,
            ),
            RandomSwap(
                local_search = gmm_shrunk,
                verbose = verbose,
                max_iterations = max_iterations,
                max_iterations_without_improvement = max_iterations_without_improvement,
            ),
            GeneticAlgorithm(
                local_search = gmm_shrunk,
                verbose = verbose,
                max_iterations = max_iterations,
                max_iterations_without_improvement = max_iterations_without_improvement,
            ),
            # GMM OAS
            gmm_oas,
            MultiStart(
                local_search = gmm_oas,
                verbose = verbose,
                max_iterations = max_iterations,
            ),
            RandomSwap(
                local_search = gmm_oas,
                verbose = verbose,
                max_iterations = max_iterations,
                max_iterations_without_improvement = max_iterations_without_improvement,
            ),
            GeneticAlgorithm(
                local_search = gmm_oas,
                verbose = verbose,
                max_iterations = max_iterations,
                max_iterations_without_improvement = max_iterations_without_improvement,
            ),
            # GMM LW
            gmm_lw,
            MultiStart(
                local_search = gmm_lw,
                verbose = verbose,
                max_iterations = max_iterations,
            ),
            RandomSwap(
                local_search = gmm_lw,
                verbose = verbose,
                max_iterations = max_iterations,
                max_iterations_without_improvement = max_iterations_without_improvement,
            ),
            GeneticAlgorithm(
                local_search = gmm_lw,
                verbose = verbose,
                max_iterations = max_iterations,
                max_iterations_without_improvement = max_iterations_without_improvement,
            ),
            # CLUSTERING CHAINS
            ClusteringChain(kmeans, gmm),
            ClusteringChain(gmm, kmeans),
        ]

        # @printf("\"%s\" => [", dataset)
        for (i, algorithm) in enumerate(algorithms)
            seed!(algorithm, 1)

            result =
                if algorithm == kmedoids || (hasproperty(algorithm, :local_search) && algorithm.local_search == kmedoids)
                    distances = pairwise(SqEuclidean(), data, dims = 1)
                    fit(algorithm, distances, k)
                else
                    fit(algorithm, data, k)
                end

            # @printf("%.16f,", result.objective)

            c = counts(result)
            for j in 1:k
                @test c[j] == count(==(j), result.assignments)
            end

            @test result.objective ≈ benchmark[i]

            if hasproperty(result, :objective_per_cluster)
                @test result.objective ≈ sum(result.objective_per_cluster)
            end
        end
        # @printf("],\n")
    end

    return nothing
end

reset_timer!()
test_all()
print_timer(sortby = :firstexec)

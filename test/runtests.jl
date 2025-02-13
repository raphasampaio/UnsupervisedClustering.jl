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

    @testset "silhouette score" begin
        data = [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 5.0 5.0]
        assignments = [1, 2, 2, 1, 1]
        @test UnsupervisedClustering.silhouette_score(data = data, assignments = assignments, metric = Euclidean()) ≈ 0.0

        data = [2 5; 3 4; 4 6; 8 3; 9 2; 10 5; 6 10; 7 8; 8 9]
        assignments = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        @test UnsupervisedClustering.silhouette_score(data = data, assignments = assignments, metric = Euclidean()) ≈ 0.5
    end

    exit(1)

    verbose = true
    decompose_if_fails = true
    max_iterations = 30
    max_iterations_without_improvement = 15

    datasets = Dict(
        #! format: off
        "3_2_-0.26" => [26416.7106071853340836,26399.3161993783542130,26399.3161993783542130,26399.3161993783542130,36147.5081656920665409,36147.5081656920665409,36073.5957067493218346,36073.5957067493218346,9345113.0105678215622902,26750.1476956724654883,26654.1405923047459510,26654.1405923047459510,26654.1405923047459510,36671.3480709589493927,36671.3480709589493927,36671.3480709589493927,36671.3480709589493927,-7.4702212739335767,-7.4500211230970370,-7.4489366154226495,-7.4500211230970370,-7.4725939132511821,-7.4526387237766327,-7.4533339416188955,-7.4521420085002443,-7.4722211994025169,-7.4510206920328903,-7.4493953372850443,-7.4497226376357908,-7.4722211994025187,-7.4510206920328903,-7.4493953372850426,-7.4497226376357881,-7.4635411676483958,26416.7106071854141192,],
        "3_2_-0.1" => [59601.5900345182162710,59567.5943560912564863,59567.5943560912710382,59567.5943560912564863,63170.3206470119039295,63170.3206470118966536,63170.3206470119039295,63170.3206470118966536,18987340.7112460732460022,60614.1308034227404278,60439.0608680820514564,60439.0608680820514564,60439.0608680820514564,70233.8032840920786839,65703.0048996939731296,65703.0048996939731296,65703.0048996939731296,-8.4651467479894027,-8.4102349541014298,-8.4100334768429441,-8.4101824043248463,-8.4699664124117344,-8.4115802534317705,-8.4115408216358656,-8.4115802534317705,-8.4681605617681157,-8.4107629148135796,-8.4106804571296419,-8.4105989765192977,-8.4681605617681193,-8.4107629148135814,-8.4106804571296347,-8.4105989765192977,-8.4110546385982605,59601.5900345181871671,],
        "3_2_0.01" => [75954.5837682040873915,75940.7629553540755296,75940.7629553540755296,75940.7629553540755296,89184.4377358828496654,89184.4377358828496654,89184.4377358828496654,89184.4377358828496654,19017664.5753948241472244,77146.2837541920598596,77146.2837541920598596,77146.2837541920598596,77146.2837541920598596,89633.3814931831002468,89633.3814931831002468,89633.3814931831002468,89633.3814931831002468,-8.5182026740741890,-8.5180572297950512,-8.5181243009978562,-8.5180572297950512,-8.5319332437570576,-8.5285204304284701,-8.5292615394819506,-8.5285204304284701,-8.5244412706655197,-8.5215019005106267,-8.5226478094182152,-8.5215019005106267,-8.5244412706655162,-8.5215019005106214,-8.5226478094182117,-8.5215019005106196,-8.5181717001935695,75954.5837682038400089,],
        "3_2_0.21" => [54050.7068362436402822,54050.7068362436402822,54050.7068362436402822,54050.7068362436402822,58109.7044917583043571,58109.7044917582970811,58109.7044917583043571,58109.7044917582970811,28577467.4943606853485107,190748.1609571797598619,54294.3046370888041565,54294.3046370888041565,54294.3046370888041565,58430.0555698682146613,58430.0555698682146613,58430.0555698682146613,58430.0555698682146613,-8.2263103563940216,-8.2262481992798620,-8.2262482876317655,-8.2262481992798620,-8.6237533552707273,-8.2299991802277717,-8.2301746066156909,-8.2299632663079993,-8.2270765382688573,-8.2267451448259443,-8.2267842529025579,-8.2267351688995429,-8.2270765382688538,-8.2267451448259408,-8.2267842529025579,-8.2267351688995429,-8.2263106409859788,54050.7068362438003533,],
        "3_5_-0.26" => [145171.0541117231477983,145171.0541117231186945,145171.0541117231186945,145171.0541117231186945,150284.6343761875177734,146995.5582167979737278,147717.0315828337043058,146995.5582167979737278,16613451.2244664989411831,164210.6193630892084911,156312.0316052992711775,157617.3088592314743437,156312.0316052992711775,176558.0617521929962095,168598.0438247059646528,168598.0438247059646528,168598.0438247059646528,-18.9902279628581034,-18.9894390359008867,-18.9886866180432072,-18.9889584644526543,-19.0647732702474251,-19.0568682965651384,-19.0529243169820610,-19.0524957912762467,-19.0110018656908721,-19.0084532025838726,-19.0089899499282780,-19.0069350277853388,-19.0110018656908686,-19.0084532025838797,-19.0089899499282744,-19.0069350277853459,-19.0004688481004393,145189.9506311327859294,],
        "3_5_-0.1" => [140639.1790945607644971,140633.0693809579825029,140633.0693809579825029,140633.0693809579825029,151226.4580142683407757,147724.2177938571840059,148850.7255250815651380,147724.2177938571840059,28741592.7844186127185822,167245.8189150553080253,146641.2836113125085831,146811.0744057827396318,146641.2836113125085831,155558.5605324382195249,155558.5605324382195249,155558.5605324382195249,155558.5605324382195249,-18.5691678679363257,-18.4751111674275457,-18.5667407402900153,-18.4750896652624839,-18.7486347466308345,-18.7370213294714674,-18.7377183979080577,-18.7324367890290411,-18.6485306675739366,-18.6384134032780686,-18.6337407395391992,-18.6365737875567596,-18.6485306675739366,-18.6384134032780686,-18.6337407395392027,-18.6365737875567667,-18.5707220843082830,140658.3377095227187965,],
        "3_5_0.01" => [186156.1504602948843967,186118.1492456819396466,186118.1492456819687504,186118.1492456819396466,205511.4498680336691905,204413.6498413142398931,204413.6498413142398931,204413.6498413142398931,29641666.3993386998772621,196657.4337840476073325,196389.1341025999281555,196389.1341025999281555,196389.1341025999281555,230559.1544694404001348,213592.0789615402754862,213592.0789615402754862,213592.0789615402754862,-19.8157275913082671,-19.8151937057868253,-19.8150151774982923,-19.8151129297369870,-19.8308307696539465,-19.8306611769480803,-19.8308307696539465,-19.8306611769480803,-19.8163141943443151,-19.8163141943443151,-19.8160679672704489,-19.8160072457278140,-19.8163141943443151,-19.8163141943443151,-19.8160679672704418,-19.8160072457278176,-19.8152166973165720,186156.1504602951172274,],
        "3_5_0.21" => [115218.3593000063119689,115205.1984080591209931,115205.1984080591209931,115205.1984080591209931,121752.4784170263010310,121752.4784170262864791,121752.4784170263010310,121752.4784170262864791,19819490.7155972793698311,120734.2042795864690561,120734.2042795864690561,120734.2042795864690561,120734.2042795864690561,128145.4512036660889862,128145.4512036660889862,128145.4512036660889862,128145.4512036660889862,-18.3813267993559428,-18.3813152644100271,-18.3813163090532719,-18.3813150326683896,-18.3997368285819967,-18.3996036784888339,-18.3996001823870756,-18.3995578336385321,-18.3819556412348639,-18.3819335859311934,-18.3819335532215611,-18.3819335859311899,-18.3819556412348533,-18.3819335859311863,-18.3819335532215575,-18.3819335859311934,-18.3814368484425010,115218.3593000061373459,],
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

        balanced_kmedoids = BalancedKmedoids(
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
            # BALANCED KMEDOIDS
            balanced_kmedoids,
            MultiStart(
                local_search = balanced_kmedoids,
                verbose = verbose,
                max_iterations = max_iterations,
            ),
            RandomSwap(
                local_search = balanced_kmedoids,
                verbose = verbose,
                max_iterations = max_iterations,
                max_iterations_without_improvement = max_iterations_without_improvement,
            ),
            GeneticAlgorithm(
                local_search = balanced_kmedoids,
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
                if algorithm == kmedoids ||
                   algorithm == balanced_kmedoids ||
                   (hasproperty(algorithm, :local_search) && algorithm.local_search == kmedoids) ||
                   (hasproperty(algorithm, :local_search) && algorithm.local_search == balanced_kmedoids)
                    distances = pairwise(SqEuclidean(), data, dims = 1)
                    fit(algorithm, distances, k)
                else
                    fit(algorithm, data, k)
                end

            # @printf("%.16f,", result.objective)

            result_counts = counts(result)
            for j in 1:k
                @test result_counts[j] == count(==(j), result.assignments)
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

using UnsupervisedClustering

using Aqua
using DelimitedFiles
using Distances
using LinearAlgebra
using Printf
using Random
using RegularizedCovarianceMatrices
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

function test_all()
    # println("BLAS: $(BLAS.get_config())")

    @testset "Aqua.jl" begin
        @testset "Ambiguities" begin
            Aqua.test_ambiguities(UnsupervisedClustering, recursive = false)
        end
        Aqua.test_all(UnsupervisedClustering, ambiguities = false)
    end

    @testset "n = 0" begin
        n, d, k = 0, 2, 3
        data = zeros(n, d)

        @testset "kmeans" begin
            algorithm = Kmeans(rng = MersenneTwister(1))
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
            algorithm = Kmedoids(rng = MersenneTwister(1))
            distances = pairwise(SqEuclidean(), data, dims = 1)
            result = fit(algorithm, distances, k)
            @test length(result.assignments) == 0

            result = fit(algorithm, data, Vector{Int}())
            @test length(result.assignments) == 0
        end

        @testset "gmm" begin
            algorithm = GMM(rng = MersenneTwister(1), estimator = EmpiricalCovarianceMatrix(n, d))
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
            algorithm = Kmeans(rng = MersenneTwister(1))
            @test_throws AssertionError result = fit(algorithm, data, k)
        end

        @testset "ksegmentation" begin
            algorithm = Ksegmentation()
            @test_throws AssertionError result = fit(algorithm, data, k)
        end

        @testset "gmm" begin
            algorithm = GMM(rng = MersenneTwister(1), estimator = EmpiricalCovarianceMatrix(n, d))
            @test_throws AssertionError result = fit(algorithm, data, k)
        end
    end

    @testset "k > n" begin
        n, d, k = 2, 2, 3
        data = zeros(n, d)

        @testset "kmeans" begin
            algorithm = Kmeans(rng = MersenneTwister(1))
            @test_throws AssertionError result = fit(algorithm, data, k)
        end

        @testset "ksegmentation" begin
            algorithm = Ksegmentation()
            @test_throws AssertionError result = fit(algorithm, data, k)
        end

        @testset "kmedoids" begin
            algorithm = Kmedoids(rng = MersenneTwister(1))
            distances = pairwise(SqEuclidean(), data, dims = 1)
            @test_throws AssertionError result = fit(algorithm, distances, k)
        end

        @testset "gmm" begin
            algorithm = GMM(rng = MersenneTwister(1), estimator = EmpiricalCovarianceMatrix(n, d))
            @test_throws AssertionError result = fit(algorithm, data, k)
        end
    end

    @testset "n = k" begin
        n, d, k = 3, 2, 3
        data = rand(MersenneTwister(1), n, d)

        @testset "kmeans" begin
            algorithm = Kmeans(rng = MersenneTwister(1))
            result = fit(algorithm, data, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end

        @testset "ksegmentation" begin
            algorithm = Ksegmentation()
            result = fit(algorithm, data, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end

        @testset "kmedoids" begin
            algorithm = Kmedoids(rng = MersenneTwister(1))
            distances = pairwise(SqEuclidean(), data, dims = 1)
            result = fit(algorithm, distances, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end

        @testset "gmm" begin
            algorithm = GMM(rng = MersenneTwister(1), estimator = EmpiricalCovarianceMatrix(n, d))
            result = fit(algorithm, data, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end
    end

    @testset "n = k (zeros)" begin
        n, d, k = 3, 2, 3
        data = zeros(n, d)

        @testset "kmeans" begin
            algorithm = Kmeans(rng = MersenneTwister(1))
            result = fit(algorithm, data, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end

        @testset "kmedoids" begin
            algorithm = Kmedoids(rng = MersenneTwister(1))
            distances = pairwise(SqEuclidean(), data, dims = 1)
            result = fit(algorithm, distances, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end

        @testset "gmm" begin
            algorithm = GMM(rng = MersenneTwister(1), estimator = EmpiricalCovarianceMatrix(n, d))
            result = fit(algorithm, data, k)
            @test sort(result.assignments) == [i for i in 1:k]
        end
    end

    @testset "convert" begin
        kmeans_result = UnsupervisedClustering.KmeansResult([1, 2, 2], [1.0 2.0; 1.0 2.0; 1.0 2.0])
        gmm_result = UnsupervisedClustering.GMMResult([1, 2, 2], [0.5, 0.5], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [Symmetric(Matrix{Float64}(I, 3, 3)) for _ in 1:2])

        @test kmeans_result == convert(UnsupervisedClustering.KmeansResult, gmm_result)
        @test gmm_result == convert(UnsupervisedClustering.GMMResult, kmeans_result)
    end

    @testset "concatenate" begin
        @test_throws MethodError concatenate()

        @testset "kmeans" begin
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

        @testset "ksegmentation" begin
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

        @testset "kmedoids" begin
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
        @testset "kmeans" begin
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

        @testset "kmedoids" begin
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
    max_iterations = 30
    max_iterations_without_improvement = 15

    datasets = Dict(
        #! format: off
        "3_2_-0.26" => [26407.1103530476975720,26399.3161993783432990,26399.3161993783432990,26399.3161993783432990,9345113.0105678215622902,26780.3459995037919725,26654.1405923047605029,26654.1405923047605029,26654.1405923047605029,-7.4571097765729348,-7.4511810399623668,-7.4487976201493487,-7.4494239964159741,-7.4604467501006217,-7.4540267497805965,-7.4513603000955575,-7.4521388252831233,-7.4589528681144266,-7.4518618170544340,-7.4491836940705181,-7.4498596924980758,-7.4589528681144293,-7.4518618170544366,-7.4491836940705181,-7.4498596924980740,-7.4625400975165652,26409.1452182245047879,],
        "3_2_-0.1" => [59601.5900345181798912,59567.5943560912564863,59567.5943560912564863,59567.5943560912564863,18987340.7112460732460022,60614.1308034227404278,60439.0608680820660084,60439.0608680820660084,60439.0608680820660084,-8.4657768527013157,-8.4105421695124551,-8.4102606292561575,-8.4105421695124551,-8.4168258492913282,-8.4120020219516309,-8.4116174720086203,-8.4116186894871809,-8.4687151993652332,-8.4107944254578211,-8.4105986361961609,-8.4105427138020872,-8.4687151993652279,-8.4107944254578193,-8.4105986361961591,-8.4105427138020872,-8.4110546385982605,59601.5900345183472382,],
        "3_2_0.01" => [75954.5837682041164953,75940.7629553540900815,75940.7629553540900815,75940.7629553540900815,19017664.5753948241472244,77192.4973030580586055,77146.2837541921180673,77146.2837541921180673,77146.2837541921180673,-8.5183676928976446,-8.5181583585887566,-8.5181206850852860,-8.5180169095136087,-8.5296338772328966,-8.5285577496525971,-8.5287792508837050,-8.5285577496525971,-8.5220302460731912,-8.5215986124306013,-8.5216359471140048,-8.5215986124306049,-8.5220302460731947,-8.5215986124306031,-8.5216359471140102,-8.5215986124305960,-8.5181717001935731,75954.5837682038691128,],
        "3_2_0.21" => [54050.7068362436621101,54050.7068362436621101,54050.7068362436621101,54050.7068362436621101,28577467.4943606853485107,54294.3046370887896046,54294.3046370887896046,54294.3046370887896046,54294.3046370887896046,-8.2263554114406823,-8.2262561008658039,-8.2262563917332461,-8.2262535568873574,-8.2299405955459122,-8.2296196310883314,-8.2299405955459122,-8.2296196310883314,-8.2268377649704707,-8.2267390189824248,-8.2268377649704671,-8.2267390189824212,-8.2268377649704654,-8.2267390189824212,-8.2268377649704689,-8.2267390189824248,-8.2263106409859823,54050.7068362437421456,],
        "3_5_-0.26" => [145184.5732566549559124,145171.0541117230604868,145171.0541117230604868,145171.0541117230604868,16613451.2244664989411831,158207.3925174557953142,156312.0316052992711775,156312.0316052992711775,156312.0316052992711775,-18.9891872029963373,-18.9891872029963373,-18.9886850580836146,-18.9877385022533964,-19.0552504950977237,-19.0552504950977237,-19.0543489820912733,-19.0546068330319471,-19.0113626318509041,-19.0090347038900518,-19.0101179816980697,-19.0090347038900518,-19.0113626318508935,-19.0090347038900518,-19.0101179816980661,-19.0090347038900447,-19.0005118813971343,145189.9506311327277217,],
        "3_5_-0.1" => [140661.0843438301235437,140633.0693809580116067,140639.1790945607062895,140633.0693809580116067,28741592.7844186127185822,147865.0032843176741153,146976.6989344926550984,146641.2836113125958946,146811.0744057828269433,-18.5708903589591792,-18.5670828184130237,-18.5667603390274714,-18.5667164270224383,-18.7461900458923445,-18.7390034800007115,-18.7342968140589825,-18.7336042450848907,-18.6429012335511395,-18.6386205208667093,-18.6349931121808332,-18.6335593802822501,-18.6429012335511359,-18.6386205208667093,-18.6349931121808297,-18.6335593802822466,-18.5706857866084505,140658.3377095221949276,],
        "3_5_0.01" => [186156.1504602948843967,186118.1492456819978543,186118.1492456819978543,186118.1492456819978543,29641666.3993386998772621,196537.5143761900253594,196389.1341025999281555,196389.1341025999281555,196389.1341025999281555,-19.8160034262380336,-19.8150985461044051,-19.8150447222830586,-19.8150985461044051,-19.8319037039006965,-19.8306811362665911,-19.8308893920641047,-19.8306811362665911,-19.8168075716168133,-19.8161933111008715,-19.8159293271795995,-19.8160221001722547,-19.8168075716168133,-19.8161933111008715,-19.8159293271795995,-19.8160221001722583,-19.8152166973165684,186156.1504602949426044,],
        "3_5_0.21" => [115218.3593000062537612,115205.1984080591937527,115205.1984080591937527,115205.1984080591937527,19819490.7155972793698311,120734.2042795864981599,120734.2042795864981599,120734.2042795864981599,120734.2042795864981599,-18.8363976495247520,-18.3813155177239729,-18.3813162791367084,-18.3813156605617181,-18.3995251905144492,-18.3995251905144492,-18.3995251905144492,-18.3995251905144492,-18.8410201940922946,-18.3819355051377613,-18.3819316557629300,-18.3819357119094917,-18.8410201940922981,-18.3819355051377578,-18.3819316557629264,-18.3819357119094882,-18.3814368484424975,115218.3593000060791383,],
        #! format: on
    )

    for (dataset, benchmark) in datasets
        data, k = get_data(dataset)
        n, d = size(data)

        kmeans = Kmeans(
            verbose = verbose,
            rng = MersenneTwister(1),
        )

        ksegmentation = Ksegmentation()

        kmedoids = Kmedoids(
            verbose = verbose,
            rng = MersenneTwister(1),
        )

        gmm = GMM(
            verbose = verbose,
            rng = MersenneTwister(1),
            estimator = EmpiricalCovarianceMatrix(n, d),
            decompose_if_fails = false,
        )

        gmm_shrunk = GMM(
            verbose = verbose,
            rng = MersenneTwister(1),
            estimator = ShrunkCovarianceMatrix(n, d),
            decompose_if_fails = false,
        )

        gmm_oas = GMM(
            verbose = verbose,
            rng = MersenneTwister(1),
            estimator = LedoitWolfCovarianceMatrix(n, d),
            decompose_if_fails = false,
        )

        gmm_lw = GMM(
            verbose = verbose,
            rng = MersenneTwister(1),
            estimator = LedoitWolfCovarianceMatrix(n, d),
            decompose_if_fails = false,
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

            # @test result.objective ≈ sum(result.objective_per_cluster)
        end
        # @printf("],\n")
    end

    return nothing
end

reset_timer!()
test_all()
print_timer(sortby = :firstexec)
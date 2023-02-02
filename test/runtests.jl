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
    println("BLAS: $(BLAS.get_config())")

    @testset "Aqua.jl" begin
        @testset "Ambiguities" begin
            Aqua.test_ambiguities(UnsupervisedClustering, recursive = false)
        end
        Aqua.test_all(UnsupervisedClustering, ambiguities = false)
    end

    @testset "n = 0" begin
        n, d, k = 0, 2, 3
        data = zeros(n, d)

        kmeans = Kmeans(rng = MersenneTwister(1))
        result = UnsupervisedClustering.fit(kmeans, data, k)
        @test length(result.assignments) == 0
    
        kmedoids = Kmedoids(rng = MersenneTwister(1))
        result = UnsupervisedClustering.fit(kmedoids, data, k)
        @test length(result.assignments) == 0
    
        gmm = GMM(rng = MersenneTwister(1), estimator = EmpiricalCovarianceMatrix(n, d))
        result = UnsupervisedClustering.fit(gmm, data, k)
        @test length(result.assignments) == 0
    end

    @testset "d = 0" begin
        n, d, k = 3, 0, 3
        data = zeros(n, d)

        kmeans = Kmeans(rng = MersenneTwister(1))
        @test_throws AssertionError result = UnsupervisedClustering.fit(kmeans, data, k)
    
        kmedoids = Kmedoids(rng = MersenneTwister(1))
        @test_throws AssertionError result = UnsupervisedClustering.fit(kmedoids, data, k)
    
        gmm = GMM(rng = MersenneTwister(1), estimator = EmpiricalCovarianceMatrix(n, d))
        @test_throws AssertionError result = UnsupervisedClustering.fit(gmm, data, k)
    end   

    @testset "k > n" begin
        n, d, k = 2, 2, 3
        data = zeros(n, d)

        kmeans = Kmeans(rng = MersenneTwister(1))
        @test_throws AssertionError result = UnsupervisedClustering.fit(kmeans, data, k)
    
        kmedoids = Kmedoids(rng = MersenneTwister(1))
        @test_throws AssertionError result = UnsupervisedClustering.fit(kmedoids, data, k)
    
        gmm = GMM(rng = MersenneTwister(1), estimator = EmpiricalCovarianceMatrix(n, d))
        @test_throws AssertionError result = UnsupervisedClustering.fit(gmm, data, k)
    end  

    @testset "n = k" begin
        n, d, k = 3, 2, 3
        data = rand(MersenneTwister(1), n, d)

        kmeans = Kmeans(rng = MersenneTwister(1))
        result = UnsupervisedClustering.fit(kmeans, data, k)
        @test sort(result.assignments) == [i for i in 1:k]
    
        kmedoids = Kmedoids(rng = MersenneTwister(1))
        result = UnsupervisedClustering.fit(kmedoids, data, k)
        @test sort(result.assignments) == [i for i in 1:k]
    
        gmm = GMM(rng = MersenneTwister(1), estimator = EmpiricalCovarianceMatrix(n, d))
        result = UnsupervisedClustering.fit(gmm, data, k)
        @test sort(result.assignments) == [i for i in 1:k]
    end   

    verbose = true
    max_iterations = 30
    max_iterations_without_improvement = 15

    datasets = Dict(
        "3_5_-0.26_1" => [145184.5732566549559124,145171.0541117230604868,145171.0541117230604868,145171.0541117230604868,158207.3925174557953142,156312.0316052992711775,156312.0316052992711775,156312.0316052992711775,-18.9891872029963373,-18.9891872029963373,-18.9886850580836182,-18.9877385022534000,-19.0552504950977237,-19.0552504950977237,-19.0543489820912768,-19.0546068330319471,-19.0113626318509041,-19.0090347038900518,-19.0101179816980732,-19.0090347038900518,-19.0113626318508970,-19.0090347038900518,-19.0101179816980661,-19.0090347038900518,],
        "3_10_-0.1_1" => [268221.2839061952545308,268172.9332773343194276,268172.9332773343194276,268172.9332773343194276,306392.2279977733269334,306392.2279977733269334,306392.2279977733269334,306392.2279977733269334,-36.0701386143378215,-36.0700546524963741,-35.6252695877579413,-35.7236672187880586,-36.1526701287235213,-36.1511744835207338,-36.1507420450296166,-36.1506079461837047,-36.0861990517536881,-36.0844894011338297,-36.0842773821205540,-36.0841655690720486,-36.0861990517537095,-36.0844894011338297,-36.0842773821205611,-36.0841655690720486,],
        "3_20_-0.1_1" => [716376.8968991155270487,716125.9244638500967994,716102.1900860873283818,716116.9963649909477681,855809.9187766498653218,855809.9187766498653218,855809.9187766498653218,855809.9187766498653218,-74.0541731390447637,-71.2662420989042289,-69.4456100221810999,-69.5146704986111104,-73.6392003586746000,-73.6019374412733924,-73.5730322112243016,-73.5730514011866745,-73.6633193884080839,-73.6130593978235197,-73.5914035543991076,-73.5916792368731763,-73.6633193884080981,-73.6130593978235055,-73.5914035543991361,-73.5916792368731763,],
        "3_2_0.01_1" => [75954.5837682041164953,75940.7629553540900815,75940.7629553540900815,75940.7629553540900815,77192.4973030580586055,77146.2837541921180673,77146.2837541921180673,77146.2837541921180673,-8.5183676928976411,-8.5181583585887566,-8.5181206850852860,-8.5180169095136051,-8.5296338772328966,-8.5285577496525953,-8.5287792508837050,-8.5285577496525953,-8.5220302460731912,-8.5215986124306013,-8.5216359471140102,-8.5215986124306031,-8.5220302460731947,-8.5215986124306049,-8.5216359471140120,-8.5215986124306085,],
        "3_10_0.01_1" => [301283.3591094072326086,301237.3496410869993269,301237.3496410869993269,301237.3496410869993269,387784.6496470365091227,334993.1816868183086626,334993.1816868183086626,334993.1816868183086626,-37.0521150235601695,-36.7092899245386661,-36.3700156548310503,-36.3513958536779853,-37.1732171359381667,-36.7939313876001890,-36.7938766980795791,-36.7939272711147467,-37.1039677763858293,-36.7223545823646305,-36.7237491339272495,-36.7223545823646305,-37.1039677763858364,-36.7223545823646305,-36.7237491339272495,-36.7223545823646376,],
        "3_5_0.21_1" => [115218.3593000062537612,115205.1984080591937527,115205.1984080591937527,115205.1984080591937527,120734.2042795864981599,120734.2042795864981599,120734.2042795864981599,120734.2042795864981599,-18.8363976495247520,-18.3813155177239729,-18.3813162791367084,-18.3813156605617145,-18.3995251905144492,-18.3995251905144492,-18.3995251905144492,-18.3995251905144492,-18.8410201940922946,-18.3819355051377578,-18.3819316557629300,-18.3819357119094988,-18.8410201940922981,-18.3819355051377613,-18.3819316557629335,-18.3819357119094953,],
        "3_10_0.21_1" => [352671.5722019832464866,352671.5722019832464866,352671.5722019832464866,352671.5722019832464866,664439.7471119252732024,438730.2076053911587223,438730.2076053911587223,438730.2076053911587223,-38.7046904358686135,-37.8080185761139589,-37.6792263631342763,-37.6447921549881386,-38.7622849297212966,-37.9048539915559743,-37.9048522487918120,-37.9048523307295468,-38.7075303072421946,-37.8124984109952749,-37.8124993828906284,-37.8124984109952678,-38.7075303072421875,-37.8124984109952749,-37.8124993828906284,-37.8124984109952891,],
        "3_20_-0.26_1" => [627893.4720151649089530,621496.1675309385173023,621534.4175140098668635,621496.1675309385173023,794116.6897856049472466,747702.4164905140642077,760630.2414160150801763,747702.4164905140642077,-70.0885372948167742,-68.6698046721149353,-67.7731261864210666,-68.6698046721149353,-73.1614346192940275,-73.0128010334429547,-72.9550419463998594,-72.9561140976563820,-73.2446384121676033,-73.1190463877579475,-73.2446384121676033,-73.1052342505084312,-73.2446384121676033,-73.1190463877579475,-73.2446384121675891,-73.1052342505084312,],
        "3_10_-0.26_1" => [338371.1996784181101248,336711.1604426605044864,337343.4663073366391473,336711.1604426605044864,386067.5309692462324165,374664.0003522558836266,376243.2435601762263104,374664.0003522558836266,-37.6124483938289558,-36.9305733790929693,-36.8801601565592989,-36.8586027949297588,-37.6744409662091684,-37.6744409662091684,-37.6661279667148463,-37.6696381291666640,-37.7596885745012543,-37.6609918816022216,-37.6331268780515558,-37.6148650495822494,-37.7596885745012614,-37.6609918816022287,-37.6331268780515558,-37.6148650495822423,],
        "3_20_0.21_1" => [574657.8239689305191860,574657.8239689305191860,574657.8239689305191860,574657.8239689305191860,763209.1386499848449603,762234.0950123307993636,762234.0950123307993636,762234.0950123307993636,-72.1520656261879196,-70.2272467300641665,-68.2524240963042530,-69.2984321722777423,-71.4610721871376740,-71.4610713606333121,-71.4610619178101842,-71.4610611280368602,-71.8890955309392723,-71.1059606114097846,-71.1059604590188172,-71.1059606050834674,-71.8890955309392723,-71.1059606114097846,-71.1059604590188172,-71.1059606050834532,],
        "3_5_0.01_1" => [186156.1504602948843967,186118.1492456819978543,186118.1492456819978543,186118.1492456819978543,196537.5143761900253594,196389.1341025999281555,196389.1341025999281555,196389.1341025999281555,-19.8160034262380336,-19.8150985461044051,-19.8150447222830586,-19.8150985461044051,-19.8319037039007000,-19.8306811362665911,-19.8308893920641047,-19.8306811362665911,-19.8168075716168062,-19.8161933111008750,-19.8159293271795995,-19.8160221001722547,-19.8168075716168097,-19.8161933111008715,-19.8159293271795995,-19.8160221001722583,],
        "3_20_0.01_1" => [1072075.8804722724016756,1070150.6831239967141300,1070030.3400313435122371,1070027.2639505288098007,1343359.4675552865955979,1332547.4219892320688814,1326012.2123062580358237,1332547.4219892320688814,-76.5717092677514160,-73.6507626103151409,-73.6144808699160222,-73.6049942570044209,-76.2125131158242368,-76.1329317117132547,-76.1329524186302535,-76.1329317117132547,-76.9365741050414158,-76.1233158249595903,-76.7284704749161506,-76.1233158249596187,-76.9365741050414300,-76.1233158249596187,-76.7284704749161648,-76.1233158249596045,],
        "3_5_-0.1_1" => [270409.2003043714212254,267607.4271683744736947,267607.4271683744736947,267607.4271683744736947,302996.9947583366301842,283495.7735194113338366,283495.7735194113338366,283495.7735194113338366,-20.4407678145330891,-20.3407130477588680,-20.3405135072101118,-20.2081102946181304,-20.4675360028264386,-20.3551886542710285,-20.3549653048172026,-20.3548830044256022,-20.4757644346218264,-20.3462478894139345,-20.3463259127097835,-20.3462478894139487,-20.4757644346218264,-20.3462478894139451,-20.3463259127097764,-20.3462478894139451,],
        "3_2_-0.1_1" => [59601.5900345181798912,59567.5943560912564863,59567.5943560912564863,59567.5943560912564863,60614.1308034227404278,60439.0608680820660084,60439.0608680820660084,60439.0608680820660084,-8.4657768527013157,-8.4105421695124569,-8.4102606292561575,-8.4105421695124569,-8.4168258492913282,-8.4120020219516327,-8.4116174720086185,-8.4116186894871809,-8.4687151993652314,-8.4107944254578229,-8.4105986361961609,-8.4105427138020890,-8.4687151993652332,-8.4107944254578229,-8.4105986361961644,-8.4105427138020890,],
        "3_2_-0.26_1" => [26407.1103530476975720,26399.3161993783432990,26399.3161993783432990,26399.3161993783432990,26780.3459995037919725,26654.1405923047605029,26654.1405923047605029,26654.1405923047605029,-7.4571097765729339,-7.4511810399623668,-7.4487976201493487,-7.4494239964159732,-7.4604467501006217,-7.4540267497805965,-7.4513603000955575,-7.4521388252831233,-7.4589528681144275,-7.4518618170544366,-7.4491836940705189,-7.4498596924980758,-7.4589528681144293,-7.4518618170544366,-7.4491836940705207,-7.4498596924980713,],
        "3_2_0.21_1" => [54050.7068362436621101,54050.7068362436621101,54050.7068362436621101,54050.7068362436621101,54294.3046370887968806,54294.3046370887968806,54294.3046370887968806,54294.3046370887968806,-8.2263554114406823,-8.2262561008658039,-8.2262563917332461,-8.2262535568873574,-8.2299405955459122,-8.2296196310883314,-8.2299405955459122,-8.2296196310883314,-8.2268377649704707,-8.2267390189824212,-8.2268377649704689,-8.2267390189824230,-8.2268377649704707,-8.2267390189824265,-8.2268377649704671,-8.2267390189824248,],
    )

    for (dataset, benchmark) in datasets
        data, k = get_data(dataset)
        n, d = size(data)

        kmeans = Kmeans(
            verbose = verbose, 
            rng = MersenneTwister(1),
        )

        kmedoids = Kmedoids(
            verbose = verbose, 
            rng = MersenneTwister(1),
        )

        gmm = GMM(
            verbose = verbose, 
            rng = MersenneTwister(1),
            estimator = EmpiricalCovarianceMatrix(n, d),
        )

        gmm_shrunk = GMM(
            verbose = verbose, 
            rng = MersenneTwister(1),
            estimator = ShrunkCovarianceMatrix(n, d),
        )

        gmm_oas = GMM(
            verbose = verbose, 
            rng = MersenneTwister(1),
            estimator = LedoitWolfCovarianceMatrix(n, d),
        )

        gmm_lw = GMM(
            verbose = verbose, 
            rng = MersenneTwister(1),
            estimator = LedoitWolfCovarianceMatrix(n, d),
        )

        algorithms = [
            # KMEANS
            kmeans,
            MultiStart(local_search = kmeans, verbose = verbose, max_iterations = max_iterations),
            RandomSwap(local_search = kmeans, verbose = verbose, max_iterations = max_iterations, max_iterations_without_improvement = max_iterations_without_improvement),
            GeneticAlgorithm(local_search = kmeans, verbose = verbose, max_iterations = max_iterations, max_iterations_without_improvement = max_iterations_without_improvement),
            # KMEDOIDS
            kmedoids,
            MultiStart(local_search = kmedoids, verbose = verbose, max_iterations = max_iterations),
            RandomSwap(local_search = kmedoids, verbose = verbose, max_iterations = max_iterations, max_iterations_without_improvement = max_iterations_without_improvement),
            GeneticAlgorithm(local_search = kmedoids, verbose = verbose, max_iterations = max_iterations, max_iterations_without_improvement = max_iterations_without_improvement),
            # GMM EMPIRICAL
            gmm,
            MultiStart(local_search = gmm, max_iterations = max_iterations),
            RandomSwap(local_search = gmm, verbose = verbose, max_iterations = max_iterations, max_iterations_without_improvement = max_iterations_without_improvement),
            GeneticAlgorithm(local_search = gmm, verbose = verbose, max_iterations = max_iterations, max_iterations_without_improvement = max_iterations_without_improvement),
            # # GMM SHRUNK
            gmm_shrunk,
            MultiStart(local_search = gmm_shrunk, max_iterations = max_iterations),
            RandomSwap(local_search = gmm_shrunk, verbose = verbose, max_iterations = max_iterations, max_iterations_without_improvement = max_iterations_without_improvement),
            GeneticAlgorithm(local_search = gmm_shrunk, verbose = verbose, max_iterations = max_iterations, max_iterations_without_improvement = max_iterations_without_improvement),
            # GMM OAS
            gmm_oas,
            MultiStart(local_search = gmm_oas, verbose = verbose, max_iterations = max_iterations),
            RandomSwap(local_search = gmm_oas, verbose = verbose, max_iterations = max_iterations, max_iterations_without_improvement = max_iterations_without_improvement),
            GeneticAlgorithm(local_search = gmm_oas, verbose = verbose, max_iterations = max_iterations, max_iterations_without_improvement = max_iterations_without_improvement),
            # GMM LW
            gmm_lw,
            MultiStart(local_search = gmm_lw, verbose = verbose, max_iterations = max_iterations),
            RandomSwap(local_search = gmm_lw, verbose = verbose, max_iterations = max_iterations, max_iterations_without_improvement = max_iterations_without_improvement),
            GeneticAlgorithm(local_search = gmm_lw, verbose = verbose, max_iterations = max_iterations, max_iterations_without_improvement = max_iterations_without_improvement)
        ]

        # @printf("\"%s\" => [", dataset)
        for (i, algorithm) in enumerate(algorithms)
            @info "$dataset, $i"
            UnsupervisedClustering.seed!(algorithm, 1)
            result = UnsupervisedClustering.fit(algorithm, data, k)
            # @printf("%.16f,", result.objective)

            counts = UnsupervisedClustering.counts(result)
            for j in 1:k
                @test counts[j] == count(==(j), result.assignments)
            end

            @static if v"1.8" <= VERSION && VERSION < v"1.9"
                @test result.objective ≈ benchmark[i]
            else
                @test result.objective ≈ benchmark[i] skip = true
            end
        end
        # @printf("],\n")
    end

    return nothing
end

reset_timer!()
@testset "UnsupervisedClustering" test_all()
print_timer(sortby = :firstexec)

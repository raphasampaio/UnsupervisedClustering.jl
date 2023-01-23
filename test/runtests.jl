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

    reset_timer!()

    @testset "Aqua.jl" begin
        @testset "Ambiguities" begin
            Aqua.test_ambiguities(UnsupervisedClustering, recursive = false)
        end
        Aqua.test_all(UnsupervisedClustering, ambiguities = false)
    end

    verbose = true
    max_iterations = 10
    max_iterations_without_improvement = 5

    datasets = Dict(
        "3_5_-0.26_1" => [145184.5732566549559124,145171.0541117230604868,145171.0541117230604868,145171.0541117230604868,158207.3925174557953142,156312.0316052992711775,156312.0316052992711775,156312.0316052992711775,-18.9891872029963373,-18.9891872029963373,-18.9886850580836182,-18.9889729987348694,-19.0552504950977237,-19.0552504950977237,-19.0552504950977237,-19.0546068330319471,-19.0113626318509041,-19.0113626318509041,-19.0101179816980661,-19.0090347038900518,-19.0113626318509041,-19.0113626318509077,-19.0101179816980661,-19.0090347038900518,],
        "3_10_-0.1_1" => [268221.2839061952545308,268184.0001413833815604,268173.6523971196147613,268172.9332773343194276,306392.2279977733269334,306392.2279977733269334,306392.2279977733269334,306392.2279977733269334,-36.0701386143378215,-36.0701386143378215,-35.6258625374700912,-35.7236672187880586,-36.1526701287235213,-36.1520518773746815,-36.1507599980806944,-36.1506079461837047,-36.0861990517536881,-36.0861990517536952,-36.0843203386667568,-36.0841655690720486,-36.0861990517537023,-36.0861990517537023,-36.0843203386667639,-36.0841655690720486,],
        "3_20_-0.1_1" => [716376.8968991155270487,716125.9244638500967994,716116.9963649909477681,716116.9963649909477681,855809.9187766498653218,855809.9187766498653218,855809.9187766498653218,855809.9187766498653218,-74.0541731390447637,-71.2662420989042289,-69.4847064507821415,-71.0663248839974244,-73.6392003586746000,-73.6093262366485277,-73.5730322112243016,-73.5730514011866745,-73.6633193884080839,-73.6130593978235197,-73.5914035543991076,-73.5916792368731763,-73.6633193884080839,-73.6130593978235197,-73.5914035543991218,-73.5916792368731763,],
        "3_2_0.01_1" => [75954.5837682041164953,75940.7629553540900815,75940.7629553540900815,75940.7629553540900815,77192.4973030580586055,77146.2837541921180673,77146.2837541921180673,77146.2837541921180673,-8.5183676928976411,-8.5181583585887566,-8.5181206850852860,-8.5180169095136051,-8.5296338772328966,-8.5296338772328966,-8.5287792508837050,-8.5285577496525953,-8.5220302460731912,-8.5220302460731912,-8.5218301831482020,-8.5215986124305996,-8.5220302460731912,-8.5220302460731929,-8.5218301831482002,-8.5215986124306013,],
        "3_10_0.01_1" => [301283.3591094072326086,301237.3496410869993269,301237.3496410869993269,301237.3496410869993269,387784.6496470365091227,334993.1816868183086626,334993.1816868183086626,334993.1816868183086626,-37.0521150235601695,-36.7122269188666053,-36.3700156548310503,-36.3513958536779853,-37.1732171359381667,-36.7940602435829334,-36.7938766980795791,-36.7939313876001890,-37.1039677763858293,-36.7225769180335888,-36.7293203049574402,-36.7223545823646162,-37.1039677763858151,-36.7225769180335888,-36.7293203049574402,-36.7223545823646162,],
        "3_5_0.21_1" => [115218.3593000062537612,115205.1984080591937527,115205.1984080591937527,115205.1984080591937527,120734.2042795864981599,120734.2042795864981599,120734.2042795864981599,120734.2042795864981599,-18.8363976495247520,-18.3813155177239729,-18.3813163533227950,-18.3813155795142684,-18.3995251905144492,-18.3995251905144492,-18.3995251905144492,-18.3995251905144492,-18.8410201940922946,-18.3819355051377578,-18.3819343790859477,-18.3819355051377613,-18.8410201940922981,-18.3819355051377578,-18.3819343790859477,-18.3819355051377578,],
        "3_10_0.21_1" => [352671.5722019832464866,352671.5722019832464866,352671.5722019832464866,352671.5722019832464866,664439.7471119252732024,438730.2076053911587223,438730.2076053911587223,438730.2076053911587223,-38.7046904358686135,-37.8080187037651001,-37.6792263746668681,-37.8080188407393152,-38.7622849297212966,-37.9048539915559743,-37.9048522487918120,-37.9048523307295468,-38.7075303072421946,-37.8124998169832196,-38.4065807411084208,-37.8124984109952749,-38.7075303072421946,-37.8124998169832125,-38.4065807411084137,-37.8124984109952678,],
        "3_20_-0.26_1" => [627893.4720151649089530,621496.1675309385173023,621534.4175140098668635,621496.1675309385173023,794116.6897856049472466,747702.4164905140642077,769898.0102130207233131,747702.4164905140642077,-70.0885372948167742,-70.0885372948167742,-70.0263705267939258,-68.6698046721149353,-73.1614346192940275,-73.0824966505202127,-72.9586703921646347,-72.9699258952851011,-73.2446384121676033,-73.2446384121676033,-73.2446384121676033,-73.1052342505084312,-73.2446384121676033,-73.2446384121676033,-73.2446384121676033,-73.1052342505084454,],
        "3_10_-0.26_1" => [338371.1996784181101248,336997.8081002151593566,337343.4663073366391473,336711.1604426605044864,386067.5309692462324165,377998.2291963486932218,376243.2435601762263104,374664.0003522558836266,-37.6124483938289558,-36.9305733790929693,-37.0089083980023403,-36.9160485733018717,-37.6744409662091684,-37.6744409662091684,-37.6736276570519522,-37.6696381291666640,-37.7596885745012543,-37.6689966159825644,-37.7372193670412130,-37.6277595558474616,-37.7596885745012898,-37.6689966159825573,-37.7372193670412130,-37.6277595558474758,],
        "3_20_0.21_1" => [574657.8239689305191860,574657.8239689305191860,574657.8239689305191860,574657.8239689305191860,763209.1386499848449603,762234.0950123307993636,763209.1386499848449603,762234.0950123307993636,-72.1520656261879196,-70.9867202754855242,-69.1986824874628894,-69.2984321722777423,-71.4610721871376740,-71.4610713606333121,-71.4610721871376740,-71.4610719012418372,-71.8890955309392723,-71.1261649015994664,-71.1157313576358519,-71.1059605783928106,-71.8890955309392723,-71.1261649015994664,-71.1157313576358661,-71.1059605783928248,],
        "3_5_0.01_1" => [186156.1504602948843967,186118.1492456819978543,186118.1492456819978543,186118.1492456819978543,196537.5143761900253594,196389.1341025999281555,196389.1341025999281555,196389.1341025999281555,-19.8160034262380336,-19.8154361559691665,-19.8150520959512804,-19.8150985461044051,-19.8319037039007000,-19.8306811362665911,-19.8310721887559502,-19.8306811362665911,-19.8168075716168062,-19.8161933111008750,-19.8159725661377841,-19.8161933111008750,-19.8168075716168133,-19.8161933111008750,-19.8159725661377806,-19.8161933111008715,],
        "3_20_0.01_1" => [1072075.8804722724016756,1070150.6831239967141300,1070030.3400313435122371,1070027.2639505288098007,1343359.4675552865955979,1343359.4675552865955979,1332848.0965741125401109,1332547.4219892320688814,-76.5717092677514160,-75.4079107519208804,-74.2573579172665745,-73.6049942570044209,-76.2125131158242368,-76.1401595012380881,-76.1330453029049465,-76.1329317117132547,-76.9365741050414158,-76.1248651664672025,-76.9365741050414158,-76.1233158249596045,-76.9365741050414158,-76.1248651664671883,-76.9365741050414300,-76.1233158249596045,],
        "3_5_-0.1_1" => [270409.2003043714212254,267890.1466974714421667,267607.4271683744736947,267607.4271683744736947,302996.9947583366301842,283495.7735194113338366,293069.1337012322619557,283495.7735194113338366,-20.4407678145330891,-20.3418459747556355,-20.4348542899572898,-20.3406490085278087,-20.4675360028264386,-20.3551886542710285,-20.4619301805432059,-20.3548830044256022,-20.4757644346218264,-20.3462478894139345,-20.4668040686535022,-20.3462478894139380,-20.4757644346218264,-20.3462478894139416,-20.4668040686535022,-20.3462478894139416,],
        "3_2_-0.1_1" => [59601.5900345181798912,59567.5943560912564863,59569.6459844519267790,59567.5943560912564863,60614.1308034227404278,60439.0608680820660084,60439.0608680820660084,60439.0608680820660084,-8.4657768527013157,-8.4107498931429152,-8.4102606292561575,-8.4105421695124569,-8.4168258492913282,-8.4120020219516327,-8.4124172239303672,-8.4116186894871809,-8.4687151993652314,-8.4108151454998463,-8.4114170256622387,-8.4107944254578229,-8.4687151993652314,-8.4108151454998481,-8.4114170256622405,-8.4107944254578211,],
        "3_2_-0.26_1" => [26407.1103530476975720,26399.3161993783432990,26399.7904201285418821,26399.3161993783432990,26780.3459995037919725,26654.1405923047605029,26750.1476956724618503,26654.1405923047605029,-7.4571097765729339,-7.4571097765729339,-7.4507262999655346,-7.4494239964159732,-7.4604467501006217,-7.4604467501006217,-7.4543382622038887,-7.4521388252831233,-7.4589528681144275,-7.4589528681144275,-7.4514978839388899,-7.4498596924980758,-7.4589528681144319,-7.4589528681144319,-7.4514978839388926,-7.4498596924980740,],
        "3_2_0.21_1" => [54050.7068362436621101,54050.7068362436621101,54050.7068362436621101,54050.7068362436621101,54294.3046370887968806,54294.3046370887968806,54294.3046370887968806,54294.3046370887968806,-8.2263554114406823,-8.2262610244959653,-8.2262563917332461,-8.2262535568873574,-8.2299405955459122,-8.2299405955459122,-8.2299405955459122,-8.2296196310883314,-8.2268377649704707,-8.2268141770120486,-8.2268377649704707,-8.2267390189824248,-8.2268377649704707,-8.2268141770120451,-8.2268377649704689,-8.2267390189824265,],
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

            @static if v"1.8" <= VERSION && VERSION < v"1.9"
                @test result.objective ≈ benchmark[i]
            else
                @test result.objective ≈ benchmark[i] skip = true
            end
        end
        # @printf("],\n")
    end

    print_timer(sortby = :firstexec)

    return nothing
end

test_all()

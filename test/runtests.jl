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
    max_iterations = 50

    datasets = Dict(
        "3_5_-0.26_1" => [154957.1261776057945099,145171.0541117230604868,145171.0541117230604868,145171.0541117230604868,170734.2684244557458442,156312.0316052992711775,156312.0316052992711775,156312.0316052992711775,-19.0388542112063419,-18.8368644624302455,-19.0169349386126108,-18.8368644624302455,-19.1028506710508275,-19.0542753967013816,-19.0520714903802961,-19.0542753967013816,-19.0597497790508115,-19.0095671957549648,-19.0055811176914418,-19.0095671957549719,-19.0597497790508079,-19.0095671957549790,-19.0055811176914489,-19.0095671957549790,],
        "3_10_-0.1_1" => [268204.0762298569898121,268172.9332773343194276,268172.9332773343194276,268172.9332773343194276,325950.5801660600118339,306392.2279977733269334,306392.2279977733269334,306392.2279977733269334,-36.0703183590430427,-35.0132869326047000,-35.3259809763222279,-35.0132869326047000,-36.1509377329519950,-36.1506991551319672,-36.1505877122244428,-36.1506991551319672,-36.0842675133474842,-36.0841703601010124,-36.0841481516946772,-36.0841703601010124,-36.0842675133474842,-36.0841703601010124,-36.0841481516946772,-36.0841703601010124,],
        "3_20_-0.1_1" => [716442.5211556990398094,716102.1900860873283818,716125.9244638500967994,716102.1900860873283818,936671.0275533118983731,855809.9187766498653218,865580.9173373164376244,855809.9187766498653218,-74.0838279747994051,-69.9351378756134352,-69.4483498533536050,-69.6338109112899701,-74.2084104047727635,-73.5730360817370581,-73.5729372327739100,-73.5730360817370581,-74.2882471888903666,-73.5917310390301083,-74.0372017304865722,-73.5917310390300941,-74.2882471888903524,-73.5917310390300798,-74.0372017304865722,-73.5917310390300798,],
        "3_2_0.01_1" => [75954.5837682041164953,75940.7629553540900815,75940.7629553540900815,75940.7629553540900815,77146.2837541921180673,77146.2837541921180673,77146.2837541921180673,77146.2837541921180673,-8.5182010741651464,-8.5180777807917956,-8.5180740045178815,-8.5180489046924261,-8.5303361286472104,-8.5284001098557525,-8.5287564556900577,-8.5284001098557525,-8.5224293187677915,-8.5214578069181197,-8.5218122487398116,-8.5214578069181215,-8.5224293187677915,-8.5214578069181108,-8.5218122487398063,-8.5214578069181162,],
        "3_10_0.01_1" => [301283.3591094072326086,301237.3496410869993269,301237.3496410869993269,301237.3496410869993269,334993.1816868183086626,334993.1816868183086626,334993.1816868183086626,334993.1816868183086626,-36.7163337885938503,-36.5121470153317276,-36.3355472757378450,-36.4164820061602583,-36.7938458384438647,-36.7938159992672169,-36.7938045955299486,-36.7938159992672169,-36.7309917532060410,-36.7223091530504817,-36.7220261274322795,-36.7220430961797604,-36.7309917532060481,-36.7223091530504746,-36.7220261274322866,-36.7220430961797533,],
        "3_5_0.21_1" => [115214.8964348844310734,115205.1984080591937527,115205.1984080591937527,115205.1984080591937527,120734.2042795864981599,120734.2042795864981599,120734.2042795864981599,120734.2042795864981599,-18.3813180045590627,-18.3813151801255295,-18.3813170123998013,-18.3813151801255295,-18.3998258408313404,-18.3994949425985332,-18.3995389415453268,-18.3994949425985332,-18.3819490618048427,-18.3819326542921253,-18.3819304744531244,-18.3819273834290620,-18.3819490618048533,-18.3819326542921289,-18.3819304744531209,-18.3819273834290549,],
        "3_10_0.21_1" => [352671.5722019832464866,352671.5722019832464866,352671.5722019832464866,352671.5722019832464866,438730.2076053911587223,438730.2076053911587223,438730.2076053911587223,438730.2076053911587223,-37.8081674247661468,-37.8080185698620213,-37.8080186195913868,-37.8080186183076847,-37.9048639684855857,-37.9048508841333316,-37.9048505569886203,-37.9048517775397613,-37.8125607242538777,-37.8124984261002979,-37.8124989488110046,-37.8124992754621871,-37.8125607242538777,-37.8124984261003121,-37.8124989488110188,-37.8124992754622014,],
        "3_20_-0.26_1" => [622007.9246746967546642,622007.9246746967546642,621548.6363981807371601,621510.9083132183877751,785298.8572100675664842,747702.4164905140642077,750297.2594066881574690,747702.4164905140642077,-73.2924242356142344,-70.4911635680603723,-67.7723935460449809,-68.0137440113894058,-73.0261462058717115,-72.9326872640375257,-72.8476015576255094,-72.8807147720406050,-73.6087451916373823,-73.2113537692527672,-73.1061868171655362,-73.1719356934801937,-73.6087451916373112,-73.2113537692527672,-73.1061868171655220,-73.1719356934801937,],
        "3_10_-0.26_1" => [336947.4319684674264863,336748.1363485666806810,336706.0805883291177452,336711.1604426605044864,381080.1344687577220611,371521.2336056475178339,379764.5464144810102880,371521.2336056475178339,-37.6586282576445228,-37.0663062777715879,-36.4545806656294857,-36.8639107095339966,-37.7115219945635118,-37.6684680837383041,-37.6697310257743183,-37.6524544722028764,-37.7459075116256884,-37.6929087692574285,-37.6221941155238682,-37.6834897081565074,-37.7459075116256884,-37.6929087692574072,-37.6221941155238753,-37.6834897081565217,],
        "3_20_0.21_1" => [687937.3633403787389398,574657.8239689305191860,574657.8239689305191860,574657.8239689305191860,878969.7630383457290009,762234.0950123307993636,762234.0950123307993636,762234.0950123307993636,-72.6467244560655132,-69.7368802690264005,-67.8456929631603174,-69.3454025753721055,-72.9171836688065582,-71.4610709487135125,-71.4610663562089883,-71.4610617030276813,-72.7275301299009840,-71.1059605431080826,-71.9227179114756865,-71.1059606306784246,-72.7275301299009698,-71.1059605431080826,-71.9227179114756865,-71.1059606306784246,],
        "3_5_0.01_1" => [186156.1504602948843967,186118.1492456819978543,186118.1492456819978543,186118.1492456819978543,253687.1892723691998981,196389.1341025999281555,196389.1341025999281555,196389.1341025999281555,-19.8168685933069781,-19.8150275517706582,-19.8149994844554058,-19.8150275517706582,-19.8333402065971569,-19.8306004502690918,-19.8307142247670569,-19.8304999064768630,-19.8174966975787399,-19.8159613446582767,-19.8160383430205442,-19.8159613446582767,-19.8174966975787434,-19.8159613446582838,-19.8160383430205584,-19.8159613446582803,],
        "3_20_0.01_1" => [1070883.7245843189302832,1070027.2639505288098007,1070027.2639505288098007,1070027.2639505288098007,1377489.1952040879987180,1324563.6066203259397298,1322707.5743514578789473,1324563.6066203259397298,-77.0737574418236875,-72.7987379879754002,-73.5942321388627505,-72.5878088434300111,-76.1407495253517084,-76.1348423930671885,-76.1329621558143259,-76.1332002295671941,-76.2264520918900672,-76.1257002883629355,-76.1233153917286103,-76.1240371322783034,-76.2264520918900672,-76.1257002883629355,-76.1233153917286103,-76.1240371322783034,],
        "3_5_-0.1_1" => [275487.8841434057103470,267607.4271683744736947,267607.4271683744736947,267607.4271683744736947,322233.2297199374879710,283495.7735194113338366,283495.7735194113338366,283495.7735194113338366,-20.4349068112379193,-20.1663988365183400,-20.3404853617050030,-20.1663988365183400,-20.3891289410647332,-20.3549306448531873,-20.3548984191709046,-20.3550297856087390,-20.3823698522648300,-20.3462537610411296,-20.3460612030407688,-20.3461989871467637,-20.3823698522648264,-20.3462537610411225,-20.3460612030407617,-20.3461989871467530,],
        "3_2_-0.1_1" => [59601.5900345181798912,59567.5943560912564863,59567.5943560912564863,59567.5943560912564863,60439.0608680820660084,60439.0608680820660084,60439.0608680820660084,60439.0608680820660084,-8.4129801329858616,-8.4107289647537780,-8.4102992040947804,-8.4102932386494498,-8.4134043055056758,-8.4117347213294362,-8.4118129516095479,-8.4115748798663699,-8.4125172640617656,-8.4109712481218839,-8.4108218616092092,-8.4105979312971488,-8.4125172640617638,-8.4109712481218892,-8.4108218616092127,-8.4105979312971577,],
        "3_2_-0.26_1" => [26426.4752462552605721,26399.3161993783432990,26399.3161993783432990,26399.3161993783432990,32029.8684217827067187,26654.1405923047605029,26654.1405923047605029,26654.1405923047605029,-7.4669181592793761,-7.4495655146955819,-7.4483027733369482,-7.4484904797151934,-7.4698241250226767,-7.4525016417215832,-7.4512318851167203,-7.4519451675227693,-7.4674296693871653,-7.4500406457634245,-7.4488021103966702,-7.4493342839995016,-7.4674296693871591,-7.4500406457634183,-7.4488021103966711,-7.4493342839994980,],
        "3_2_0.21_1" => [54050.7068362436621101,54050.7068362436621101,54050.7068362436621101,54050.7068362436621101,54294.3046370887968806,54294.3046370887968806,54294.3046370887968806,54294.3046370887968806,-8.2262549104633234,-8.2262549104633234,-8.2262542879134166,-8.2262497652243329,-8.2303749097041141,-8.2299269129342569,-8.2299766587398970,-8.2299269129342569,-8.2268582216710495,-8.2267398630209954,-8.2268020513250946,-8.2267015554266134,-8.2268582216710513,-8.2267398630209954,-8.2268020513250981,-8.2267015554266187,],
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
            RandomSwap(local_search = kmeans, verbose = verbose, max_iterations = max_iterations),
            GeneticAlgorithm(local_search = kmeans, verbose = verbose, max_iterations = max_iterations),
            # KMEDOIDS
            kmedoids,
            MultiStart(local_search = kmedoids, verbose = verbose, max_iterations = max_iterations),
            RandomSwap(local_search = kmedoids, verbose = verbose, max_iterations = max_iterations),
            GeneticAlgorithm(local_search = kmedoids, verbose = verbose, max_iterations = max_iterations),
            # GMM EMPIRICAL
            gmm,
            MultiStart(local_search = gmm, max_iterations = max_iterations),
            RandomSwap(local_search = gmm, verbose = verbose, max_iterations = max_iterations),
            GeneticAlgorithm(local_search = gmm, verbose = verbose, max_iterations = max_iterations),
            # # GMM SHRUNK
            gmm_shrunk,
            MultiStart(local_search = gmm_shrunk, max_iterations = max_iterations),
            RandomSwap(local_search = gmm_shrunk, verbose = verbose, max_iterations = max_iterations),
            GeneticAlgorithm(local_search = gmm_shrunk, verbose = verbose, max_iterations = max_iterations),
            # GMM OAS
            gmm_oas,
            MultiStart(local_search = gmm_oas, verbose = verbose, max_iterations = max_iterations),
            RandomSwap(local_search = gmm_oas, verbose = verbose, max_iterations = max_iterations),
            GeneticAlgorithm(local_search = gmm_oas, verbose = verbose, max_iterations = max_iterations),
            # GMM LW
            gmm_lw,
            MultiStart(local_search = gmm_lw, verbose = verbose, max_iterations = max_iterations),
            RandomSwap(local_search = gmm_lw, verbose = verbose, max_iterations = max_iterations),
            GeneticAlgorithm(local_search = gmm_lw, verbose = verbose, max_iterations = max_iterations),
        ]

        # @printf("\"%s\" => [", dataset)
        for (i, algorithm) in enumerate(algorithms)
            UnsupervisedClustering.seed!(algorithm, 1)
            result = UnsupervisedClustering.fit(algorithm, data, k)
            # @printf("%.16f,", result.objective)

            @static if VERSION <= v"1.8"
                @test result.objective ≈ benchmark[i]
            else
                @test result.objective ≈ benchmark[i] skip = true
            end
        end
        # @printf("],\n")
    end

    print_timer(sortby = :firstexec)

    return
end

test_all()

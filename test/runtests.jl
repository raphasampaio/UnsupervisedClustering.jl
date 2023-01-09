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
        @testset "ambiguities" begin
            Aqua.test_ambiguities(UnsupervisedClustering, recursive = false)
        end
        Aqua.test_all(UnsupervisedClustering, ambiguities = false)
    end

    verbose = false

    datasets = Dict(
        "3_5_-0.26_1" => [154957.1261776057945099,145171.0541117230604868,145171.0541117230604868,145171.0541117230604868,170734.2684244557458442,156312.0316052992711775,156312.0316052992711775,156312.0316052992711775,-19.0388542112063419,-18.9804361168857092,-18.9884454528302768,-18.7548839891544752,-19.1028506710508275,-19.0532506533084245,-19.0520714903802961,-19.0531310852587268,-19.0597497790508150,-19.0085782040028626,-19.0055811176914418,-19.0051500822511059,-19.0597497790508044,-19.0085782040028697,-19.0055811176914453,-19.0051500822511095,],
        "3_10_-0.1_1" => [268204.0762298569898121,268172.9332773343194276,268172.9332773343194276,268172.9332773343194276,325950.5801660600118339,306392.2279977733269334,306392.2279977733269334,306392.2279977733269334,-36.0703183590430427,-35.8057508653141667,-34.6562540747107164,-35.3578866398683402,-36.1509377329520021,-36.1505595153009551,-36.1503935532764658,-36.1506991551319672,-36.0842675133474842,-36.0841406229986887,-36.0841481516946914,-36.0841325669944268,-36.0842675133474771,-36.0841406229986745,-36.0841481516946985,-36.0841325669944197,],
        "3_20_-0.1_1" => [716442.5211556990398094,716102.1900860873283818,716116.9963649909477681,716102.1900860873283818,936671.0275533118983731,855809.9187766498653218,855809.9187766498653218,855809.9187766498653218,-74.0838279747994335,-69.9351378750553891,-69.2984036432075783,-69.3263738421643865,-74.2084104047727635,-73.5730360817370581,-73.5729289283472383,-73.5729571328639196,-74.2882471888903666,-73.5917310390301083,-74.0103306250959747,-73.5911660346668128,-74.2882471888903666,-73.5917310390300941,-74.0103306250959889,-73.5911660346668128,],
        "3_2_0.01_1" => [75954.5837682041164953,75940.7629553540900815,75940.7629553540900815,75940.7629553540900815,77146.2837541921180673,77146.2837541921180673,77146.2837541921180673,77146.2837541921180673,-8.5182010741651464,-8.5180695452986246,-8.5179945671259851,-8.5180195914592307,-8.5303361286472086,-8.5284001098557525,-8.5284964329609814,-8.5284001098557525,-8.5224293187677933,-8.5214578069181179,-8.5215839382927054,-8.5214578069181108,-8.5224293187677951,-8.5214578069181091,-8.5215839382926966,-8.5214578069181073,],
        "3_10_0.01_1" => [301283.3591094072326086,301237.3496410869993269,301237.3496410869993269,301237.3496410869993269,334993.1816868183086626,334993.1816868183086626,334993.1816868183086626,334993.1816868183086626,-36.7163337885938503,-36.3695289587291342,-36.3064552436141739,-36.3310955619628615,-36.7938458384438647,-36.7937844420940223,-36.7937668053230880,-36.7938159992672169,-36.7309917532060410,-36.7221282839724452,-36.7220172921167887,-36.7220257423311693,-36.7309917532060410,-36.7221282839724665,-36.7220172921168100,-36.7220257423311764,],
        "3_5_0.21_1" => [115214.8964348844310734,115205.1984080591937527,115205.1984080591937527,115205.1984080591937527,120734.2042795864981599,120734.2042795864981599,120734.2042795864981599,120734.2042795864981599,-18.3813180045590663,-18.3813150632202316,-18.3813149727706957,-18.3813151801255295,-18.3998258408313440,-18.3994949425985297,-18.3995389415453232,-18.3994949425985297,-18.3819490618048462,-18.3819245532406867,-18.3819286816425524,-18.3819273834290691,-18.3819490618048604,-18.3819245532406903,-18.3819286816425382,-18.3819273834290691,],
        "3_10_0.21_1" => [352671.5722019832464866,352671.5722019832464866,352671.5722019832464866,352671.5722019832464866,438730.2076053911587223,438730.2076053911587223,438730.2076053911587223,438730.2076053911587223,-37.8081674247661539,-37.3843330709346660,-37.6348986944804409,-37.6989344788726513,-37.9048639684855857,-37.9048480603149187,-37.9048505569886274,-37.9048517775397613,-37.8125607242538848,-37.8124983759565225,-37.8124989488110117,-37.8124993144694272,-37.8125607242538848,-37.8124983759565154,-37.8124989488110046,-37.8124993144694272,],
        "3_20_-0.26_1" => [622007.9246746967546642,621695.1380325681529939,621546.1170849310001358,621510.9083132183877751,785298.8572100675664842,747702.4164905140642077,750297.2594066881574690,747702.4164905140642077,-73.2924242356142344,-69.0596726970445758,-67.7651941283640298,-67.8147069680011469,-73.0261462058717257,-72.9095062944623464,-72.8476015576255094,-72.7779192420338745,-73.6087451916373823,-73.0460482463311394,-73.1049243269147269,-73.0653393833928959,-73.6087451916373112,-73.0460482463311536,-73.1049243269147411,-73.0653393833928959,],
        "3_10_-0.26_1" => [336947.4319684674264863,336748.1363485666806810,336694.4416484995745122,336683.5811427895096131,381080.1344687577220611,371521.2336056475178339,372915.9188730129972100,371521.2336056475178339,-37.6586282576445228,-36.9077737446954899,-36.1797314278213094,-36.4712644697482986,-37.7115219945635118,-37.6547302061985363,-37.6659613196237970,-37.6521812809327940,-37.7459075116256813,-37.6304392960076086,-37.6056899271404745,-37.6360389084265421,-37.7459075116256884,-37.6304392960076086,-37.6056899271404816,-37.6360389084265492,],
        "3_20_0.21_1" => [687937.3633403787389398,574657.8239689305191860,574657.8239689305191860,574657.8239689305191860,878969.7630383457290009,762234.0950123307993636,762234.0950123307993636,762234.0950123307993636,-72.6467244560655132,-69.3145173256776559,-67.7789318668468894,-67.8672827851901133,-72.9171836688065582,-71.4610703591285414,-71.4610614896789116,-71.4610617030276813,-72.7275301299009840,-71.1059605431080826,-71.8402227852306510,-71.1059606270353584,-72.7275301299009698,-71.1059605431080541,-71.8402227852306652,-71.1059606270353726,],
        "3_5_0.01_1" => [186156.1504602948843967,186118.1492456819978543,186118.1492456819978543,186118.1492456819978543,253687.1892723691998981,196389.1341025999281555,196389.1341025999281555,196389.1341025999281555,-19.8168685933069781,-19.8150275517706582,-19.8149994844554058,-19.8150275517706582,-19.8333402065971569,-19.8304446961183736,-19.8304475443417161,-19.8304999064768701,-19.8174966975787399,-19.8159343890310353,-19.8159799900275750,-19.8159613446582767,-19.8174966975787328,-19.8159343890310282,-19.8159799900275786,-19.8159613446582732,],
        "3_20_0.01_1" => [1070883.7245843189302832,1070027.2639505288098007,1070027.2639505288098007,1070027.2639505288098007,1377489.1952040879987180,1322856.7126853447407484,1319611.5028715548105538,1319611.5028715548105538,-77.0737574418236875,-72.7987379926280056,-73.5941916024261786,-72.7227594425491048,-76.1407495253517084,-76.1348151745531823,-76.1329351264368199,-76.1329481574661742,-76.2264520918900672,-76.1257002883629355,-76.1233153917286103,-76.1232827114588133,-76.2264520918900672,-76.1257002883629212,-76.1233153917286103,-76.1232827114587991,],
        "3_5_-0.1_1" => [275487.8841434057103470,267607.4271683744736947,267607.4271683744736947,267607.4271683744736947,322233.2297199374879710,283495.7735194113338366,283495.7735194113338366,283495.7735194113338366,-20.4349068112379229,-20.2523865091876694,-20.3404022473277735,-20.2131701537173925,-20.3891289410647332,-20.3549306448531873,-20.3547753355716594,-20.3548885634430903,-20.3823698522648336,-20.3462308760394208,-20.3460612030407724,-20.3461989871467637,-20.3823698522648300,-20.3462308760394173,-20.3460612030407688,-20.3461989871467672,],
        "3_2_-0.1_1" => [59601.5900345181798912,59567.5943560912564863,59567.5943560912564863,59567.5943560912564863,60439.0608680820660084,60439.0608680820660084,60439.0608680820660084,60439.0608680820660084,-8.4129801329858633,-8.4100742244496427,-8.4100174714808258,-8.4099366090680689,-8.4134043055056758,-8.4114234980061706,-8.4114980684277771,-8.4115748798663681,-8.4125172640617674,-8.4107229931017393,-8.4105243754837531,-8.4105907588185627,-8.4125172640617709,-8.4107229931017411,-8.4105243754837513,-8.4105907588185609,],
        "3_2_-0.26_1" => [26426.4752462552605721,26399.3161993783432990,26399.3161993783432990,26399.3161993783432990,32029.8684217827067187,26654.1405923047605029,26654.1405923047605029,26654.1405923047605029,-7.4669181592793743,-7.4494816549227938,-7.4482634783382657,-7.4483885678501318,-7.4698241250226767,-7.4520829068685925,-7.4512318851167185,-7.4511838164108983,-7.4674296693871653,-7.4496009822932194,-7.4488021103966764,-7.4490703702205909,-7.4674296693871653,-7.4496009822932256,-7.4488021103966791,-7.4490703702206016,],
        "3_2_0.21_1" => [54050.7068362436621101,54050.7068362436621101,54050.7068362436621101,54050.7068362436621101,54294.3046370887968806,54294.3046370887968806,54294.3046370887968806,54294.3046370887968806,-8.2262549104633234,-8.2262468195781722,-8.2262463074227643,-8.2262457841869097,-8.2303749097041141,-8.2297365822164590,-8.2299766587398970,-8.2299269129342569,-8.2268582216710495,-8.2267398630209954,-8.2267955374119719,-8.2267015554266116,-8.2268582216710477,-8.2267398630209918,-8.2267955374119701,-8.2267015554266205,],
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
            MultiStart(local_search = kmeans, verbose = verbose),
            RandomSwap(local_search = kmeans, verbose = verbose),
            GeneticAlgorithm(local_search = kmeans, verbose = verbose),
            # KMEDOIDS
            kmedoids,
            MultiStart(local_search = kmedoids, verbose = verbose),
            RandomSwap(local_search = kmedoids, verbose = verbose),
            GeneticAlgorithm(local_search = kmedoids, verbose = verbose),
            # GMM EMPIRICAL
            gmm,
            MultiStart(local_search = gmm),
            RandomSwap(local_search = gmm, verbose = verbose),
            GeneticAlgorithm(local_search = gmm, verbose = verbose),
            # # GMM SHRUNK
            gmm_shrunk,
            MultiStart(local_search = gmm_shrunk),
            RandomSwap(local_search = gmm_shrunk, verbose = verbose),
            GeneticAlgorithm(local_search = gmm_shrunk, verbose = verbose),
            # GMM OAS
            gmm_oas,
            MultiStart(local_search = gmm_oas, verbose = verbose),
            RandomSwap(local_search = gmm_oas, verbose = verbose),
            GeneticAlgorithm(local_search = gmm_oas, verbose = verbose),
            # GMM LW
            gmm_lw,
            MultiStart(local_search = gmm_lw, verbose = verbose),
            RandomSwap(local_search = gmm_lw, verbose = verbose),
            GeneticAlgorithm(local_search = gmm_lw, verbose = verbose),
        ]

        @printf("\"%s\" => [", dataset)
        for (i, algorithm) in enumerate(algorithms)
            UnsupervisedClustering.seed!(algorithm, 1)
            result = UnsupervisedClustering.fit(algorithm, data, k)
            @printf("%.16f,", result.objective)

            # @static if VERSION >= v"1.8.0"
            #     @test result.objective ≈ benchmark[i]
            # end
        end
        @printf("],\n")
    end

    print_timer(sortby = :firstexec)

    return
end

test_all()

using DelimitedFiles
using Distances
using LinearAlgebra
using Printf
using Random
using RegularizedCovarianceMatrices
using Test
using TimerOutputs
using UnsupervisedClustering

# @sk_import mixture:GaussianMixture
# @sk_import cluster:KMeans
# include("gmmsk.jl")

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
    reset_timer!()

    verbose = false

    datasets = Dict(
        "3_5_-0.26_1" => [145416.1637348171789199,145171.0541117230604868,145171.0541117230604868,145171.0541117230604868,175104.8045569069508929,156312.0316052992711775,156312.0316052992711775,156312.0316052992711775,-18.9899381199827104,-18.8245776983214874,-18.9874276186482156,-18.9875145325214874,-19.0970339318041837,-19.0522360507951056,-19.0532939907053844,-19.0522360507951056,-19.0321939012966865,-19.0154727778136525,-19.0111126524558181,-19.0112474554906399,-19.0421313771965401,-19.0093724855098252,-19.0056670394616773,-19.0041537754078149,],
        "3_10_-0.1_1" => [268173.6523971196147613,268172.9332773343194276,268172.9332773343194276,268172.9332773343194276,354361.0385497465031222,306392.2279977733269334,306392.2279977733269334,306392.2279977733269334,-36.3661019940295986,-35.6392067397175225,-34.9948923952359934,-35.3425385401017138,-36.1518656291481690,-36.1503783578586209,-36.1507133300829508,-36.1504603137139782,-36.4569406396789617,-36.1164304528256750,-36.1168283396591079,-36.1164623254524884,-36.4653734804423237,-36.0841917674026149,-36.0841273339948003,-36.0839564584207011,],
        "3_20_-0.1_1" => [716372.2712337194243446,716102.1900860873283818,716102.1900860873283818,716116.9963649909477681,940850.2317491472931579,855809.9187766498653218,855809.9187766498653218,855809.9187766498653218,-74.4303846120621699,-71.0376871933657981,-71.0608636401563984,-69.1238370527535722,-73.6017918098985859,-73.5730189331536053,-73.5729397473912741,-73.5730126908632229,-74.5061589119386269,-73.6365811542582946,-73.6359148896734581,-73.6358792844683450,-73.6318163162589627,-73.5913758578526256,-73.5907758630179529,-73.5908400581837299,],
        "3_2_0.01_1" => [75940.7629553540900815,75940.7629553540900815,75940.7629553540900815,75940.7629553540900815,77192.4973030580586055,77146.2837541921180673,77146.2837541921180673,77146.2837541921180673,-8.5189517970934521,-8.5180707295423179,-8.5180391524813892,-8.5180384528618447,-8.5324014581393506,-8.5287376782903834,-8.5286773333465362,-8.5285762676834587,-8.5206118994132396,-8.5189282319454041,-8.5188408413186298,-8.5189155000454466,-8.8059515879307657,-8.5214847622874981,-8.5214232158573591,-8.5215649348146840,],
        "3_10_0.01_1" => [301283.3591094072326086,301237.3496410869993269,301237.3496410869993269,301237.3496410869993269,367877.7112739859730937,334993.1816868183086626,334993.1816868183086626,334993.1816868183086626,-36.7238108131259082,-36.3201000280413950,-36.0256133062688519,-36.3048638880661656,-36.7978130545588655,-36.7936832787009394,-36.7938487394411666,-36.7937699455139224,-36.7698943294544094,-36.7651513381213846,-36.7651483027283064,-36.7651632745412016,-36.7321103857769415,-36.7220745911647271,-36.7220220282436998,-36.7220745911647199,],
        "3_5_0.21_1" => [115205.1984080591937527,115205.1984080591937527,115205.1984080591937527,115205.1984080591937527,120734.2042795864981599,120734.2042795864981599,120734.2042795864981599,120734.2042795864981599,-18.3813164097068586,-18.3813150078458349,-18.3813149740170729,-18.3813149790065893,-18.3998324191154481,-18.3994643793244386,-18.3995119064451700,-18.3995221301222465,-18.3857989139083777,-18.3856459936730054,-18.3856656053334753,-18.3856609716327597,-18.3819530206298545,-18.3819168969888125,-18.3819274239197448,-18.3819263326617417,],
        "3_10_0.21_1" => [352671.5722019832464866,352671.5722019832464866,352671.5722019832464866,352671.5722019832464866,438730.2076053911587223,438730.2076053911587223,438730.2076053911587223,438730.2076053911587223,-37.8080213782018433,-37.8080185635501351,-37.6204485620519549,-37.7262664511606118,-37.9049412168342030,-37.9048435140012074,-37.9048487459463743,-37.9048435140012074,-37.9168704719271688,-37.9160108104867177,-37.9160684807360013,-37.9160108104867106,-37.8125258661241688,-37.8124979654693831,-37.8124988342300199,-37.8124978917905992,],
        "3_20_-0.26_1" => [624580.9059905658941716,621739.3032725828234106,621396.6251700960565358,621499.6320649372646585,779618.2331002652645111,747702.4164905140642077,747702.4164905140642077,747702.4164905140642077,-73.5621444229315955,-68.5439094052237436,-67.6975443144446274,-67.8328892444093015,-73.5549404330238445,-72.8350206665346889,-72.9260149821861319,-72.8307623268188706,-73.6159841230809491,-72.8997026727948452,-72.8482319973715136,-72.8989293501306577,-73.7492544696658427,-73.1372103854329367,-73.0736424701732687,-73.0596778982480686,],
        "3_10_-0.26_1" => [338168.0581636284478009,336753.9530302907805890,336683.5811427895096131,336694.4416484995745122,402421.8420325779588893,373536.5018435153178871,371521.2336056475178339,371521.2336056475178339,-37.9471092102180947,-36.8596395702526607,-35.9531575235748946,-35.9578030558867283,-37.7114018877043193,-37.6539294449627207,-37.6734033541956208,-37.6469930444950691,-38.0458267984985170,-37.6325105400766518,-37.5917474458284957,-37.6164138424044125,-38.0693742195511788,-37.6551189522188778,-37.6092412109109020,-37.6240986370195500,],
        "3_20_0.21_1" => [574665.0863221032777801,574657.8239689305191860,574657.8239689305191860,574657.8239689305191860,762234.0950123307993636,762234.0950123307993636,762234.0950123307993636,762234.0950123307993636,-71.0230136902979154,-69.2963470144194105,-67.8766701915200059,-67.9556866776083979,-71.4610722773393121,-71.4610707156692371,-71.4610585551125155,-71.4610619582742288,-71.5996256944239491,-71.5996198209015091,-71.5996075042731661,-71.5996052117830573,-71.1213529600853889,-71.1059605544411681,-71.1059605536928956,-71.1059606355691756,],
        "3_5_0.01_1" => [186156.1504602948843967,186118.1492456819978543,186118.1492456819978543,186118.1492456819978543,196657.4337840474909171,196389.1341025999281555,196389.1341025999281555,196389.1341025999281555,-19.8153148481742925,-19.8150439734646504,-19.8149956256706901,-19.8150349259009850,-19.8316044150954909,-19.8303875239699394,-19.8304953395263190,-19.8303875239699394,-19.8243529435648966,-19.8240452640502873,-19.8239804137222144,-19.8240209079362586,-19.8162036472594671,-19.8159893245275640,-19.8159958915989769,-19.8159563102885734,],
        "3_20_0.01_1" => [1071635.4465985286515206,1070052.9326572588179260,1070027.2639505288098007,1070027.2639505288098007,1344852.9156738263554871,1322826.3066622747574002,1325121.3167202922049910,1319611.5028715548105538,-75.7391317234233838,-73.4920748251569762,-73.6083907930966461,-72.8467964738559033,-76.1444112119659025,-76.1330241087080708,-76.1329327475775699,-76.1329408090038129,-76.2279144847382497,-76.1930091621665895,-76.1894373895873827,-76.1901045337569371,-76.1347355119923890,-76.1234036156615588,-76.1233347273680181,-76.1233152581675796,],
        "3_5_-0.1_1" => [271414.5912718095351011,267607.4271683744736947,267607.4271683744736947,267607.4271683744736947,308774.1947603798471391,283495.7735194113338366,283495.7735194113338366,283495.7735194113338366,-20.4561604763597913,-20.2142632008113381,-20.3404613201490250,-20.2877720403923476,-20.4707632583310861,-20.3549759918001953,-20.3547694308747289,-20.3547809034963123,-20.4597833201945178,-20.3521843236661866,-20.3520816629406021,-20.3519810739088136,-20.4864215159426308,-20.3462184343069303,-20.3461330883174334,-20.3462215985851884,],
        "3_2_-0.1_1" => [59567.5943560912564863,59567.5943560912564863,59567.5943560912564863,59567.5943560912564863,60439.0608680820660084,60439.0608680820660084,60439.0608680820660084,60439.0608680820660084,-8.4125083702332262,-8.4100229395501298,-8.4099342259417735,-8.4098735926722181,-8.4152294268754293,-8.4115833219737421,-8.4114500969470676,-8.4112905898013874,-8.4156551692637009,-8.4117474282232223,-8.4111989621030645,-8.4115725211077095,-8.4146120265511914,-8.4105790040089019,-8.4105534258971062,-8.4104066043763606,],
        "3_2_-0.26_1" => [26409.1452182243592688,26399.3161993783432990,26399.3161993783432990,26399.3161993783432990,26780.3459995037919725,26654.1405923047605029,26654.1405923047605029,26654.1405923047605029,-7.4794672720607274,-7.4493214198719793,-7.4484849197506389,-7.4481494801921810,-7.4622091916768261,-7.4526607937375635,-7.4512552191768764,-7.4508870847094784,-7.4783718834040265,-7.4508375912496501,-7.4494486714367483,-7.4493622499527374,-7.4610668924085708,-7.4493384795795299,-7.4490143440592069,-7.4489255335491622,],
        "3_2_0.21_1" => [54050.7068362436621101,54050.7068362436621101,54050.7068362436621101,54050.7068362436621101,54294.3046370887968806,54294.3046370887968806,54294.3046370887968806,54294.3046370887968806,-8.2263014324566193,-8.2262571327528953,-8.2262454919107455,-8.2262483248662015,-8.2303122946304175,-8.2298142734991817,-8.2299325673419510,-8.2298142734991817,-8.2271105895073866,-8.2268625620358495,-8.2269640408323408,-8.2268625620358478,-8.2268039519631984,-8.2267356747914810,-8.2267501356053572,-8.2267356747914793,],
        # "facebook_live_sellers" => [],
        # "glass" => [],
        # "handwritten_digits" => [],
        # "hcv" => [],
        # "human_activity_recognition" => [],
        # "image_segmentation" => [],
        # "ionosphere" => [],
        # "iris" => [],
        # "letter_recognition" => [],
        # "magic" => [],
        # "mice_protein" => [],
        # "pendigits" => [],
        # "scadi" => [],
        # "seeds" => [],
        # "shuttle" => [],
        # "spect" => [],
        # "waveform" => [],
        # "wholesale" => [],
        # "wines" => [],
        # "yeast" => [],
    )

    for (dataset, benchmark) in datasets
        data, k = get_data(dataset)
        n, d = size(data)

        kmeans = Kmeans(verbose = verbose)
        kmedoids = Kmedoids(verbose = verbose)
        gmm = GMM(estimator = EmpiricalCovarianceMatrix(n, d), verbose = verbose)
        gmm_shrunk = GMM(estimator = ShrunkCovarianceMatrix(n, d), verbose = verbose)
        gmm_oas = GMM(estimator =  OASCovarianceMatrix(n, d), verbose = verbose)
        gmm_lw = GMM(estimator = LedoitWolfCovarianceMatrix(n, d), verbose = verbose)

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
            GeneticAlgorithm(local_search = gmm_lw, verbose = verbose)
        ]

        # @printf("\"%s\" => [", dataset)
        for (i, algorithm) in enumerate(algorithms)
            UnsupervisedClustering.seed!(algorithm, 1)
            result = UnsupervisedClustering.fit(algorithm, data, k)
            # @printf("%.16f,", result.objective)
            if result.objective ≆ benchmark[i]
                algorithm.verbose = true
                UnsupervisedClustering.seed!(algorithm, 1)
                @show result = UnsupervisedClustering.fit(algorithm, data, k)

                @test result.objective ≈ benchmark[i]
            end
        end
        # @printf("],\n")
    end

    print_timer(sortby = :firstexec)

    return
end

test_all()

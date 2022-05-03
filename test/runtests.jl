using Test
using UnsupervisedLearning
using Random

k = 3
data = [
    15.260072562445375 15.555450517514476
    30.530585293111116 33.21252960500067
    28.755515742373284 9.50292547175411
    -20.408476267644886 -19.74653089028922
    -18.295746249176396 -21.334588118393462
    2.9157727985301705 16.31045698721363
    16.235310516679547 -28.17898854908964
    -21.31498803905579 -1.571996642209374
    -10.58112984816694 -20.900916133888074
    13.277013711113065 12.80512363841844
    14.821934640790381 6.319424906156815
    -35.191639901039665 6.790066186592135
    15.552420907539906 10.700816727572043
    34.8514594656686 -26.175643320061464
    -17.000947819389555 -2.853487019538905
    11.218625499124204 26.967822310959654
    -13.597982210888235 -4.559342452424014
    -18.601961190697143 -16.457989036653203
    29.33493242319898 20.951262333441257
    7.6819072818361995 19.047359629523868
    13.967224274294189 -41.108982342704024
    7.968127869151974 -41.76281607378729
    -22.778857650630357 -8.920676464518172
    28.44806562446375 28.44082667418016
    -26.020576034093565 26.393575750442785
    27.944851029532682 -12.933209801358231
    14.596903621841582 -4.193127047152178
    -20.783461791036167 -8.873907011690166
    28.534097222997467 -22.417164371897456
    26.29370041242507 -22.6221574744058
    18.64976954624253 21.109914879091853
    26.215334021940357 7.080517394545822
    31.78899541128091 -29.049861366595195
    -8.940221635354028 -42.08219927411801
    17.49155662036275 4.82085719186032
    -25.633166671941428 -13.445053389543256
    16.345929952187213 -25.788210148465073
    -23.87264036777467 -21.590975996950576
    -23.448294019833504 -23.413218062232833
    -17.756472895631624 -17.310082885163464
    -27.101947068730606 3.821629173674431
    8.02447111749645 14.775217380159953
    28.942980033797895 -25.182772420933322
    18.78384446874322 19.136405092925298
    30.909006676328808 -13.809454751849756
    14.407055469151745 -29.70036501414945
    43.50406248490002 -19.603116152864253
    -7.032114146511368 -16.347296625277945
    18.40697640918174 29.579162041001126
    43.52265919753152 -9.86599354436499
    -1.6175361362526566 -41.012774233027095
    58.548063805416575 -28.80107519139547
    -15.383380547298303 -6.431236723825201
    -25.00666813473383 -1.3312262801839392
    -0.8379760709081161 4.780599144553996
    18.890024518063267 -32.69601763455843
    17.47710125061208 13.724177472424481
    -1.4148817484042944 -34.24097063700426
    -11.587177936043691 -9.796696754785039
    -29.109352740929836 -17.041710215756552
    22.958273249928617 -26.106262307679327
    9.593612448628235 -39.979164580772235
    22.880611565561807 -39.460595490566355
    -8.940632975871885 -3.744039731106218
    12.624887455933449 43.255086200923756
    1.8050694870169437 3.2423266020837893
    33.58519879672686 -24.998075748136788
    -5.5124436819107 -27.66057439568604
    4.790162425160844 -29.80640414188042
    21.20449270852759 9.814387785541212
    18.412355351361924 -13.276788959521602
    -12.966074032865237 -22.737063311569177
    11.352153174648928 23.16530142127897
    51.15949912808226 -18.098218029460817
    -3.0491814279478753 -22.361793713620067
    -26.836580559833 -6.5385236716157635
    24.6446390247938 36.257120500876916
    -22.252301643847858 -18.984290315991736
    28.759250197377106 -28.235309614747166
    16.439690322647323 15.291341176871818
    21.28955905886445 10.549345547758277
    36.07664552127267 -16.849909721899497
    -16.660863997095866 -23.87468487645023
    13.682633587169953 9.753992139007817
    -6.098504513592111 3.2437091341966795
    5.791940138087906 -42.188909318301924
    -18.851029972579337 -3.1182594316868855
    38.05467575764799 -20.303935449557613
    20.672611938387814 -30.136314103234305
    8.381340058974134 -16.353346905485182
    4.4355807886415715 7.671001354489382
    30.4060607622557 35.641020802017266
    7.72711056086953 -35.70033336538316
    -2.425989190485204 -32.40666585711628
    -15.57914470039805 -38.83503116863374
    -26.830837731978242 -28.303266428991712
    -8.609559937107273 -22.925030884548256
    35.32219606253215 -30.072372507809973
    29.389091475125863 -25.53902732345067
    16.10092554836403 -30.426067310664404
    33.46329285815 -26.89249784174466
    17.484693634683367 -32.31145493402043
    -16.341812743055772 -13.265036041396877
    -12.454586743843594 -9.250674301057288
    -14.849854882704774 8.884718860731766
    19.843212323908258 -14.591447616793427
    37.70098506561318 -24.962496590170225
    12.60398545976978 22.44731989273577
    -15.118417157050072 -1.424998382003393
    26.533743407156937 27.1835898823172
    20.15284025518408 -35.178440601476645
    18.772365835241292 20.964281340038927
    33.524974193762745 40.01167064115025
    36.41003391829498 -17.738863054037036
    33.37422972588549 -20.92670660504806
    -5.875509147575354 1.1585441099199336
    12.694969834708198 35.97045852102797
    -8.67432408095528 22.481115409041436
    11.155070962761675 31.567781350523973
    15.771510895789648 -10.007112922798918
    29.285916154917345 41.13752697916587
    -5.384536861226485 -5.672578799246786
    -13.477193641539095 4.8323063108351185
    23.166829615927917 17.61128583659749
    -12.360763450414794 2.2959519408573072
    -18.281719794634636 -29.21019323411911
    7.272903838557426 5.271934081091327
    8.049739166317881 -1.0241578644507854
    12.096646189157306 -42.99014637488608
    -3.079337150101882 -20.088645807719757
    -21.424092014075477 -13.932866216959207
    -4.590724942862147 -15.41280853730503
    -12.218689607190493 -14.804993720152408
    2.37163225653093 16.051105351051
    9.720540495893239 -29.841270798889525
    20.92350118960252 -34.00974142793338
    0.8172316949152965 -40.62683734612568
    -22.49482468663342 -22.89208441056994
    25.428876515442916 -37.18821909411615
    17.826020510843016 -24.2031090698188
    27.29210059504011 -0.052146674835235274
    -21.609418058370977 -24.926496128378915
    -5.402259169667408 -17.17766563134659
    24.70753265425202 -21.79811976001836
    -2.16733266544745 -3.4334684373513653
    -10.1169659271449 -25.120009749050695
    35.336763611165495 -27.225398139058953
    49.28292522529631 -10.685720337985384
    16.70151399990762 27.43910566081103
    46.35322389896008 -7.14648261293781
    -6.586755895404611 -11.114313401440167
    20.63686333410767 -29.36696427445622
    7.338042754220112 -33.3879551035817
    11.882470771076175 -27.750664177344003
    45.37836741358572 -9.118051935551954
    22.64374798794732 -19.78066353863913
    34.26398265162739 -28.098256389396067
    21.237994171968435 28.778116596319833
    -2.514111744494688 -31.56406403985262
    24.17904335096825 9.733043127240158
    8.492676695940286 -4.89915355497698
    13.619606044726195 -14.6333184914834
    27.157719692520985 -14.27127725973219
    -15.163483673764906 -7.119877223013568
    23.279979886452946 14.394123787695001
    23.268796206461804 -27.833176151229182
    27.135431816900073 -32.229924041329504
    -25.245987035457762 -7.490988756638576
    -10.286694555869783 -7.357251285521311
    3.896422704317109 33.60609743053634
    -23.62468222036735 1.1095716617707474
    -16.104246350670905 -8.164375619583309
    16.37202613624394 39.712948280291585
    19.758549165909372 17.161366217772407
    30.138994410428964 -17.506513586448534
    -15.79477394473753 -9.934698973309619
    12.682242895590653 -35.35557338886537
    -2.518217886150282 16.65807228209755
    25.667292631407054 -26.57894492602057
    -7.982309228923934 -16.389888920522793
    11.491354703424339 19.779031585197856
    11.131460804256603 -12.031120885383189
    42.21926478439204 35.53576475459832
    -21.160534496195247 -6.401191809289356
    -10.602094175278184 -25.826333846759155
    -13.476431990493172 -13.000925787702466
    -27.89693350449185 -14.061683429622839
    -17.6780635764113 -13.490507589015841
    7.805017975785126 3.0034450233526364
    -11.83935083598309 -30.46140554907243
    18.90307156694237 -40.07600190421489
    64.68408077687855 -5.464093202868529
    -3.344837033853036 -23.488737528220714
    27.141750822131204 10.230998396512337
    28.003218922372064 -15.240806511025468
    22.329216084774913 -35.67572504141293
    -18.56291256670589 -3.4561282342081814
    10.10286244825017 2.3597047578803867
    10.059414216947028 -5.25545243502755
    14.030223194961486 25.021614347643315
    -18.888780718087027 -10.409132634489476
    4.641272634540169 23.862796019812713
    31.55140293246435 12.559649292242916
    0.5419158632157899 27.58246672675795
    -19.103394499097355 -19.843789252434206
    -0.46968855717707925 15.286485065067513
    2.7997522768625096 21.104729204582533
    30.127792980147344 19.71628442507036
    17.55604286600996 -29.69850934470391
    -13.68920055053013 -26.11181001006357
    -18.09993846960147 -37.01760946608067
    35.99670945042236 27.96553544406395
    13.589770411354376 -38.257910359609504
    -14.235005240264904 -8.258084199854224
    -4.824470582527255 -29.04947080748457
    10.859933353617576 -38.731043127483396
    6.993135137654502 -38.37234808808656
    -8.250452583374624 0.5587496667374907
    -20.89662055217235 -2.746456582614842
    -16.393105654103444 -33.262018517625165
    23.91525972534365 -35.68456013380464
    -13.915743296031303 -12.851188157133649
    26.94706640232318 10.03541496459335
    26.591246144027757 23.95904661290667
    -18.305433338807784 -5.5700836033837415
    -19.138972249057037 -15.238553546960324
    26.02959872849933 18.80006888502078
    19.318568232211817 35.003847677497326
    3.3979385927779617 4.445830135912001
    21.3390833783338 8.676453195591009
    21.724688657235152 25.449832375585196
    17.385971870582072 11.673640977193664
    58.94574238716703 -18.022630694352546
    -13.794421136022732 -21.36316821064891
    -14.466586349653392 -7.228090854203574
    23.535997960576346 31.97456018361231
    -19.735954778045446 -10.955388693885794
    -22.892542521041523 -30.826558403119655
    51.362926610969545 -9.219573545091205
    29.618793688848225 -33.203722106341175
    -7.589796010883598 -39.979056638405375
    20.940909131549798 28.339875959111893
    14.371733620024138 24.619659605315064
    -15.538684901383082 -0.031534527346774865
    24.54319384383206 -20.895043584251255
    34.95591104233635 -7.434143137696939
    -6.20832702905194 0.20347687313670804
    36.97745436940461 -23.050540784680432
    9.051087030451225 -39.66243817183644
    52.98319761431216 -21.150665439559976
    10.642559057973033 -37.354558543375745
    -18.7206601953398 24.41307402176477
    5.9770919888784615 -2.0558888182639983
    31.901038676741873 6.94106593729399
    -13.194814467960773 -20.24426079113603
    -22.183124093016495 -11.459284399718825
    20.002468495815382 31.056934831036447
    34.41236881091615 10.79039092075628
    -8.318483403538277 -46.388050252532906
    -6.097920096729322 -24.14153691620526
    27.300092727582648 3.92247009248171
    -22.684249927059604 -19.2786459775579
    1.6524785245572584 9.561747468021375
    -6.501672487115641 -29.793802702082022
    18.40688998001309 -28.012486938948104
    -18.694659644657182 -9.976279243093773
    57.54537879024369 -5.4112476307398225
    35.10492886535819 -19.600172628261873
    49.96488040510306 -4.751165757624921
    -3.890700761773383 -16.784231406357005
    -15.794083286911864 -23.7438144506528
    18.04816427480557 -38.89797018945208
    -10.127573638027073 -32.17356147238757
    19.09555196805076 18.434667681770826
    16.565317065171246 -38.40589492516557
    23.68750694436997 -31.34510876295159
    8.441138745689472 -38.0290544918235
    43.70059784614958 -6.957003287732643
    19.246108234608542 28.694928374708333
    -13.77955339667855 -20.714127743575197
    -5.5745540203227115 -30.246833319863903
    32.60257302768195 -24.968395821953596
    2.2613320702676667 -48.50934274176742
    19.911933129776834 -26.664985938183985
    12.053471804286655 19.880254423588152
    43.179590221633305 -21.73114152672457
    -14.041945566018288 -22.24794943867163
    3.3130977110211752 -34.15068189421622
    28.174030863557068 -25.526487932535204
    9.496853001767418 13.736671363707863
    -6.047926093728117 -22.753479879210623
    19.618364570038587 -33.90943809821042
    -13.917289788576023 -32.69261948882762
    28.72930767821823 12.745857941351991
    -15.54790268113403 -13.000939806944825
    8.417352142635163 0.7040796465833026
    4.14375502444749 6.833631191049504
    -0.6255706928599984 15.023088975918071
    41.07292352239909 -27.157780704744454
    27.71244569513649 -26.647196650632257
]

functions = [
    kmeans,
    kmeans_ms,
    kmeans_rs,
    kmeans_hg,
    gmm,
    gmm_shrunk,
    gmm_oas,
    gmm_ledoitwolf,
    gmm_ms,
    gmm_ms_shrunk,
    gmm_ms_oas,
    gmm_ms_ledoitwolf,
    gmm_rs,
    gmm_rs_shrunk,
    gmm_rs_oas,
    gmm_rs_ledoitwolf,
    gmm_hg,
    gmm_hg_shrunk,
    gmm_hg_oas,
    gmm_hg_ledoitwolf
]

totalcosts = [
    75954.58376820412,
    75940.76295535408,
    75940.76295535408,
    75940.76295535408,
    -8.518197227893912,
    -8.530343278829495,
    -8.519724197498986,
    -8.522435878195157,
    -8.518077892571354,
    -8.52834828693898,
    -8.518899071008791,
    -8.52143323297397,
    -8.518039743363916,
    -8.528496447382114,
    -8.518844556374447,
    -8.521812128894373,
    -8.518077892571354,
    -8.52834828693898,
    -8.518899071008791,
    -8.52143323297397
]

for i in 1:length(functions)
    Random.seed!(1)
    @test functions[i](data, k).totalcost ≈ totalcosts[i]
end

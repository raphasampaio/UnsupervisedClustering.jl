using Test
using UnsupervisedLearning

n = 64
d = 2
k = 3
data = rand(n, d)

@show kmeans(data, k)
@show kmeans_ms(data, k)
@show kmeans_rs(data, k)
@show kmeans_hg(data, k)    
@show gmm(data, k)
@show gmm_shrunk(data, k)
@show gmm_oas(data, k)
@show gmm_ledoitwolf(data, k)
@show gmm_ms(data, k)
@show gmm_ms_shrunk(data, k)
@show gmm_ms_oas(data, k)
@show gmm_ms_ledoitwolf(data, k)
@show gmm_rs(data, k)
@show gmm_rs_shrunk(data, k)
@show gmm_rs_oas(data, k)
@show gmm_rs_ledoitwolf(data, k)
@show gmm_hg(data, k)
@show gmm_hg_shrunk(data, k)
@show gmm_hg_oas(data, k)
@show gmm_hg_ledoitwolf(data, k)
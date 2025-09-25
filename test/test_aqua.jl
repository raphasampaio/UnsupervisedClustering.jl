module TestAqua

using Aqua
using UnsupervisedClustering
using Test

@testset "Aqua" begin
    Aqua.test_ambiguities(UnsupervisedClustering, recursive = false)
    Aqua.test_all(UnsupervisedClustering, ambiguities = false)
    return nothing
end

end
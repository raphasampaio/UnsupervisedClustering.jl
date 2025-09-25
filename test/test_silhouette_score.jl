module TestSilhouetteScore

using Distances
using RegularizedCovarianceMatrices
using StableRNGs
using UnsupervisedClustering
using Test

@testset "Silhouette Score" begin
    data = [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 5.0 5.0]
    assignments = [1, 2, 2, 1, 1]
    @test UnsupervisedClustering.silhouette_score(data = data, assignments = assignments, metric = Euclidean()) ≈
          0.015714285714285747

    data = [2 5; 3 4; 4 6; 8 3; 9 2; 10 5; 6 10; 7 8; 8 9]
    assignments = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    @test UnsupervisedClustering.silhouette_score(data = data, assignments = assignments, metric = Euclidean()) ≈
          0.6148550278971129

    data = [1.0 0.0; 1.0 1.0; 1.0 2.0; 2.0 3.0; 2.0 2.0; 1.0 2.0; 3.0 1.0; 3.0 3.0; 2.0 1.0]
    assignments = [1, 1, 2, 2, 2, 2, 3, 3, 3]
    @test UnsupervisedClustering.silhouette_score(data = data, assignments = assignments, metric = Euclidean()) ≈
          0.23320709938729836
end

end

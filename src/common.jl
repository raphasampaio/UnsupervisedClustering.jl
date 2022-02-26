global MAX_ITERATIONS = 10000

global METAHEURISTIC_ITERATIONS = 100

mutable struct ClusteringData
    X::Matrix{Float64}

    n::Int
	d::Int
    k::Int
    
    function ClusteringData(X::Matrix{Float64}, k::Int)
        n = size(X, 1)
        d = size(X, 2)

        return new(X, n, d, k)
    end
end

function sum_diagonal!(matrix::Matrix{Float64}, value::Float64)
    n = size(matrix, 1)

    for i in 1:n
        matrix[i, i] += value
    end
end    

function fix(matrix::Matrix{Float64}, eps::Float64)
    eigen_matrix = eigen(Symmetric(matrix))
    new_matrix = eigen_matrix.vectors * Matrix(Diagonal(max.(eigen_matrix.values, eps))) * eigen_matrix.vectors'
    return Symmetric(new_matrix)
end

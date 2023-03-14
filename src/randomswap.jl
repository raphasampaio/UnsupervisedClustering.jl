function random_swap!(result::KmeansResult, data::AbstractMatrix{<:Real}, rng::AbstractRNG)
    n, d = size(data)
    k = size(result.centers, 2)

    if n > 0 && k > 0
        to = rand(rng, 1:k)
        from = rand(rng, 1:n)
        result.centers[:, to] = copy(data[from, :])

        reset_objective!(result)
    end

    return nothing
end

function random_swap!(result::KmedoidsResult, data::AbstractMatrix{<:Real}, rng::AbstractRNG)
    n, d = size(data)
    k = length(result.centers)

    if n > 0 && k > 0
        weights = ones(n)
        for i in result.centers
            weights[i] = 0.0
        end

        to = rand(rng, 1:k)
        weights[result.centers[to]] = 1.0
        from = sample(rng, aweights(weights))
        result.centers[to] = from

        reset_objective!(result)
    end

    return nothing
end

function random_swap!(result::GMMResult, data::AbstractMatrix{<:Real}, rng::AbstractRNG)
    k = result.k
    n, d = size(data)

    if n > 0 && k > 0
        to = rand(rng, 1:k)
        from = rand(rng, 1:n)
        result.centers[to] = copy(data[from, :])

        m = mean([det(result.covariances[j]) for j in 1:k])
        value = (m > 0 ? m : 1.0)^(1 / d)
        result.covariances[to] = Symmetric(value .* Matrix{Float64}(I, d, d))

        reset_objective!(result)
    end

    return nothing
end

function fit(parameters::RandomSwap, data::AbstractMatrix{<:Real}, k::Integer)::ClusteringResult
    iterations_without_improvement = 0

    best_result = fit(parameters.local_search, data, k)

    if parameters.verbose
        print_iteration(0)
        print_iteration(iterations_without_improvement)
        print_result(best_result)
        print_string("(initial solution)")
        print_newline()
    end

    for iteration in 1:parameters.max_iterations
        result = copy(best_result)

        random_swap!(result, data, parameters.local_search.rng)

        fit!(parameters.local_search, data, result)

        if parameters.verbose
            print_iteration(iteration)
            print_iteration(iterations_without_improvement)
            print_result(result)
        end

        if isbetter(result, best_result)
            best_result = result
            iterations_without_improvement = 0

            if parameters.verbose
                print_string("(new best)")
            end
        else
            iterations_without_improvement += 1
            if iterations_without_improvement > parameters.max_iterations_without_improvement
                break
            end
        end

        if parameters.verbose
            print_newline()
        end
    end

    return best_result
end

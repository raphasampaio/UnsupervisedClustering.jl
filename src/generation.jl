mutable struct Generation
    population::Vector{Result}
    empty::Set{Int}

    function Generation()
        return new(Vector{Result}(), Set{Int}())
    end
end

function Base.println(generation::Generation)
    println("Population: ")
    for i in 1:length(generation.population)
        if in(i, generation.empty) == false
            @printf("%i - %.2f\n", i, generation.population[i].objective)
        end
    end
end

function population_size(generation::Generation)
    return length(generation.population)
end

function active_population_size(generation::Generation)
    return population_size(generation) - length(generation.empty)
end

function remove(generation::Generation, i::Int)
    reset_objective!(generation.population[i])
    push!(generation.empty, i)
    return
end

function add!(generation::Generation, result::Result)
    if length(generation.empty) > 0
        generation.population[pop!(generation.empty)] = result
    else
        push!(generation.population, result)
    end
end

function binary_tournament(generation::Generation, rng::AbstractRNG)
    size = population_size(generation)
    indices = sample(rng, 1:size, aweights([(in(i, generation.empty) ? 0 : 1) for i in 1:size]), 4, replace = false)
    parent1 = generation.population[indices[1]]
    parent2 = generation.population[indices[2]]
    parent3 = generation.population[indices[3]]
    parent4 = generation.population[indices[4]]

    return isbetter(parent1, parent2) ? parent1 : parent2, isbetter(parent3, parent4) ? parent3 : parent4
end

function crossover(parent1::Result, parent2::Result, data::AbstractMatrix{<:Real}, rng::AbstractRNG)
    k = parent1.k

    weights = zeros(k, k)
    for i in 1:k
        for j in 1:k
            weights[i, j] = distance(parent1, i, parent2, j, data)
        end
    end
    assignment, _ = hungarian(weights)

    offspring = copy(parent1)
    reset_objective!(offspring)

    for i in 1:k
        if rand(rng) > 0.5
            copy_clusters!(offspring, i, parent1, i)
        else
            copy_clusters!(offspring, i, parent2, assignment[i])
        end
    end

    update_weights!(offspring, parent1, parent2, assignment)

    return offspring
end

function eliminate(generation::Generation, to_remove::Int, rng::AbstractRNG)
    removed = 0
    size = population_size(generation)
    for i in 1:size
        if to_remove == removed
            break
        end

        for j in (i+1):(size)
            if to_remove == removed
                break
            end

            if in(i, generation.empty) == false && in(j, generation.empty) == false
                if generation.population[i].objective ≈ generation.population[j].objective
                    removed += 1
                    if rand(rng) > 0.5
                        remove(generation, i)
                    else
                        remove(generation, j)
                    end
                end
            end
        end
    end

    if to_remove > removed
        size = population_size(generation)
        threshold = partialsort(generation.population, active_population_size(generation) - (to_remove - removed), lt = isbetter)

        for i in 1:size
            if in(i, generation.empty) == false && isbetter(threshold, generation.population[i])
                remove(generation, i)
            end
        end
    end
end

function distance(a::KmeansResult, i::Int, b::KmeansResult, j::Int, data::AbstractMatrix{<:Real})
    return Distances.evaluate(Euclidean(), a.centers[:, i], b.centers[:, j])
end

function distance(a::KmedoidsResult, i::Int, b::KmedoidsResult, j::Int, data::AbstractMatrix{<:Real})
    return Distances.evaluate(Euclidean(), data[a.centers[i], :], data[b.centers[j], :])
end

function distance(a::GMMResult, i::Int, b::GMMResult, j::Int, data::AbstractMatrix{<:Real})
    distance1 = Distances.evaluate(SqMahalanobis(a.covariances[i], skipchecks = true), a.centers[i], b.centers[j])
    distance2 = Distances.evaluate(SqMahalanobis(b.covariances[j], skipchecks = true), a.centers[i], b.centers[j])
    return (distance1 + distance2) / 2
end

function copy_clusters!(destiny::KmeansResult, destiny_i::Int, source::KmeansResult, source_i::Int)
    destiny.centers[:, destiny_i] = copy(source.centers[:, source_i])
    return
end

function copy_clusters!(destiny::KmedoidsResult, destiny_i::Int, source::KmedoidsResult, source_i::Int)
    destiny.centers[destiny_i] = source.centers[source_i]
    return
end

function copy_clusters!(destiny::GMMResult, destiny_i::Int, source::GMMResult, source_i::Int)
    destiny.centers[destiny_i] = copy(source.centers[source_i])
    destiny.covariances[destiny_i] = copy(source.covariances[source_i])
    return
end

function update_weights!(child::KmeansResult, parent1::KmeansResult, parent2::KmeansResult, assignment::AbstractVector{<:Integer})
    return
end

function update_weights!(child::KmedoidsResult, parent1::KmedoidsResult, parent2::KmedoidsResult, assignment::AbstractVector{<:Integer})
    return
end

function update_weights!(child::GMMResult, parent1::GMMResult, parent2::GMMResult, assignment::AbstractVector{<:Integer})
    k = length(assignment)
    for i in 1:k
        child.weights[i] = (parent1.weights[i] + parent2.weights[assignment[i]]) / 2
        # child.weights[i] = 1 / k
    end
    return
end

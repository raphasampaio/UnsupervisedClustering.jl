mutable struct Generation{R <: AbstractResult}
    population::Vector{R}
    empty::Set{Int}

    function Generation{R}() where {R <: AbstractResult}
        return new{R}(Vector{R}(), Set{Int}())
    end
end

function population_size(generation::Generation{R}) where {R <: AbstractResult}
    return length(generation.population)
end

function active_population_size(generation::Generation{R}) where {R <: AbstractResult}
    return population_size(generation) - length(generation.empty)
end

function remove(generation::Generation{R}, i::Integer) where {R <: AbstractResult}
    reset_objective!(generation.population[i])
    push!(generation.empty, i)
    return nothing
end

function add!(generation::Generation{R}, result::R) where {R <: AbstractResult}
    if length(generation.empty) > 0
        generation.population[pop!(generation.empty)] = result
    else
        push!(generation.population, result)
    end
end

function binary_tournament(generation::Generation{R}, rng::AbstractRNG) where {R <: AbstractResult}
    size = population_size(generation)
    indices = sample(rng, 1:size, aweights([(in(i, generation.empty) ? 0 : 1) for i in 1:size]), 4, replace = false)
    parent1 = generation.population[indices[1]]
    parent2 = generation.population[indices[2]]
    parent3 = generation.population[indices[3]]
    parent4 = generation.population[indices[4]]

    return isbetter(parent1, parent2) ? parent1 : parent2, isbetter(parent3, parent4) ? parent3 : parent4
end

function get_best_solution(generation::Generation{R}) where {R <: AbstractResult}
    best_solution = generation.population[1]
    for (i, solution) in enumerate(generation.population)
        if in(i, generation.empty) == false && isbetter(solution, best_solution)
            best_solution = solution
        end
    end
    return best_solution
end

function crossover(parent1::AbstractResult, parent2::AbstractResult, data::AbstractMatrix{<:Real}, rng::AbstractRNG)
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

function eliminate(generation::Generation{R}, to_remove::Integer, rng::AbstractRNG) where {R <: AbstractResult}
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
        threshold = partialsort(
            generation.population,
            active_population_size(generation) - (to_remove - removed),
            lt = isbetter,
        )

        for i in 1:size
            if in(i, generation.empty) == false && isbetter(threshold, generation.population[i])
                remove(generation, i)
            end
        end
    end
end

function distance(a::KmeansResult, i::Integer, b::KmeansResult, j::Integer, data::AbstractMatrix{<:Real})
    return Distances.evaluate(Euclidean(), a.clusters[:, i], b.clusters[:, j])
end

function distance(a::KmedoidsResult, i::Integer, b::KmedoidsResult, j::Integer, data::AbstractMatrix{<:Real})
    c1 = a.clusters[i]
    c2 = b.clusters[j]
    return (data[c1, c2] + data[c2, c1]) / 2
end

function distance(a::GMMResult, i::Integer, b::GMMResult, j::Integer, data::AbstractMatrix{<:Real})
    d1 = Distances.evaluate(SqMahalanobis(a.covariances[i], skipchecks = true), a.clusters[i], b.clusters[j])
    d2 = Distances.evaluate(SqMahalanobis(b.covariances[j], skipchecks = true), a.clusters[i], b.clusters[j])
    return (d1 + d2) / 2
end

function copy_clusters!(destiny::KmeansResult, destiny_i::Integer, source::KmeansResult, source_i::Integer)
    destiny.clusters[:, destiny_i] = copy(source.clusters[:, source_i])
    return nothing
end

function copy_clusters!(destiny::KmedoidsResult, destiny_i::Integer, source::KmedoidsResult, source_i::Integer)
    destiny.clusters[destiny_i] = source.clusters[source_i]
    return nothing
end

function copy_clusters!(destiny::GMMResult, destiny_i::Integer, source::GMMResult, source_i::Integer)
    destiny.clusters[destiny_i] = copy(source.clusters[source_i])
    destiny.covariances[destiny_i] = copy(source.covariances[source_i])
    return nothing
end

function update_weights!(
    child::KmeansResult,
    parent1::KmeansResult,
    parent2::KmeansResult,
    assignment::AbstractVector{<:Integer},
)
    return nothing
end

function update_weights!(
    child::KmedoidsResult,
    parent1::KmedoidsResult,
    parent2::KmedoidsResult,
    assignment::AbstractVector{<:Integer},
)
    return nothing
end

function update_weights!(
    child::GMMResult,
    parent1::GMMResult,
    parent2::GMMResult,
    assignment::AbstractVector{<:Integer},
)
    k = length(assignment)
    for i in 1:k
        child.weights[i] = (parent1.weights[i] + parent2.weights[assignment[i]]) / 2
        # child.weights[i] = 1 / k
    end
    return nothing
end

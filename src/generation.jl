mutable struct Generation
    population::Vector{Result}
    empty::Set{Int}

    function Generation()
        return new(Vector{Result}(), Set{Int}())
    end
end

function Base.println(generation::Generation)
    for i in 1:length(generation.population)
        print("$(i) - ")
        if in(i, generation.empty) == false
            println(generation.population[i])
        else
            println("")
        end
    end
end

function population_size(generation::Generation)
    return length(generation.population)
end

function active_population_size(generation::Generation)
    return population_size(generation) - length(generation.empty)
end

function remove(generation, i::Int)
    invalidate!(generation.population[i])
    push!(generation.empty, i)
    return nothing
end

function add!(generation::Generation, result)
    if length(generation.empty) > 0
        generation.population[pop!(generation.empty)] = result
    else
        push!(generation.population, result)
    end
end

function add_random!(data::ClusteringData, generation::Generation, method::Function)
    result = method(data)
    add!(generation, result)
    return nothing
end

function binary_tournament(generation::Generation)
    size = population_size(generation)
    indices = sample(1:size, aweights([(in(i, generation.empty) ? 0 : 1) for i in 1:size]), 4, replace = false)
    parent1 = generation.population[indices[1]]
    parent2 = generation.population[indices[2]]
    parent3 = generation.population[indices[3]]
    parent4 = generation.population[indices[4]]

    return isbetter(parent1, parent2) ? parent1 : parent2, isbetter(parent3, parent4) ? parent3 : parent4
end

function crossover(data::ClusteringData, parent1::Result, parent2::Result)
    d = data.d
    k = data.k

    weights = zeros(k, k)
    for i in 1:k
        for j in 1:k
            weights[i, j] = distance(parent1, i, parent2, j)
        end
    end
    assignment, cost = hungarian(weights)

    offspring = copy(parent1)
    invalidate!(offspring)

    for i in 1:k
        if rand() > 0.5
            copy_clusters!(offspring, i, parent1, i)
        else
            copy_clusters!(offspring, i, parent2, assignment[i])
        end
    end

    update_weights!(offspring, parent1, parent2, assignment)

    return offspring
end

function eliminate(generation::Generation, to_remove::Int)
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
                if generation.population[i].totalcost == generation.population[j].totalcost
                    removed += 1
                    if rand() > 0.5
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

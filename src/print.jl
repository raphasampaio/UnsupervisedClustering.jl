# function total_digits(number::Integer)
#     return (number == 0) ? 1 : (log10(number) + 1)
# end

print_objective(result::ClusteringResult) = print_objective(result.objective)
function print_objective(objective::Float64)
    @printf("%12.4f ", objective)
end

function print_iteration(iteration::Integer)
    @printf("%8d ", iteration)
end

print_iterations(result::ClusteringResult) = print_iterations(result.iterations)
function print_iterations(iterations::Integer)
    @printf("%8dit ", iterations)
end

print_elapsed(result::ClusteringResult) = print_elapsed(result.elapsed)
function print_elapsed(elapsed::Float64)
    @printf("%10.2fs ", elapsed)
    return nothing
end

function print_change(change::Float64)
    @printf("%12.4f ", change)
    return nothing
end

function print_result(result::ClusteringResult)
    print_objective(result)
    print_iterations(result)
    print_elapsed(result)
    return nothing
end

function print_newline()
    @printf("\n")
    return nothing
end

function print_string(str::String)
    @printf("%s ", str)
    return nothing
end

function print_initial_centers(centers::Vector{<:Integer})
    print_string("Initial centers = [")
    for center in centers
        print_string("$center,")
    end
    print_string("]")
    print_newline()
    return nothing
end

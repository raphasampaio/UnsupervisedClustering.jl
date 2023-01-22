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
end

function print_change(change::Float64)
    @printf("%12.4f ", change)
end

function print_result(result::ClusteringResult)
    print_objective(result)
    print_iterations(result)
    print_elapsed(result)
    return
end

function print_newline()
    @printf("\n")
end

function print_string(str::String)
    @printf("%s ", str)
end

function print_initial_centers(centers::Vector{<:Integer})
    print_string("Initial centers = [")
    for i in 1:k
        print_string("$(centers[i]),")
    end
    print_string("]")
    print_newline()
end
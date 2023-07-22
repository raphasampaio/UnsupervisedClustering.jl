# function total_digits(number::Integer)
#     return (number == 0) ? 1 : (log10(number) + 1)
# end

print_objective(result::Result) = print_objective(result.objective)
function print_objective(objective::Real)
    @printf("%12.4f ", objective)
    return nothing
end

function print_iteration(iteration::Integer)
    @printf("%8d ", iteration)
    return nothing
end

print_iterations(result::Result) = print_iterations(result.iterations)
function print_iterations(iterations::Integer)
    @printf("%8dit ", iterations)
    return nothing
end

print_elapsed(result::Result) = print_elapsed(result.elapsed)
function print_elapsed(elapsed::Real)
    @printf("%10.2fs ", elapsed)
    return nothing
end

function print_change(change::Real)
    @printf("%12.4f ", change)
    return nothing
end

function print_result(result::Result)
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

function print_initial_clusters(clusters::AbstractVector{<:Integer})
    print_string("Initial clusters = [")
    for cluster in clusters
        print_string("$cluster,")
    end
    print_string("]")
    print_newline()
    return nothing
end

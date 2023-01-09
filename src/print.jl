# function total_digits(number::Integer)
#     return (number == 0) ? 1 : (log10(number) + 1)
# end

print_objective(result::Result) = print_objective(result.objective)
function print_objective(objective::Float64)
    @printf("%12.4f ", objective)
end

function print_iteration(iteration::Integer)
    @printf("%8d ", iteration)
end

print_iterations(result::Result) = print_iterations(result.iterations)
function print_iterations(iterations::Integer)
    @printf("%8dit ", iterations)
end

print_elapsed(result::Result) = print_elapsed(result.elapsed)
function print_elapsed(elapsed::Float64)
    @printf("%10.2fs ", elapsed)
end

function print_change(change::Float64)
    @printf("%12.4f ", change)
end

function print_result(result::Result)
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

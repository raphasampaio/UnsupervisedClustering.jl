using Test

function recursive_include(path::String)
    for file in readdir(path)
        file_path = joinpath(path, file)
        if isdir(file_path)
            recursive_include(file_path)
            continue
        elseif !endswith(file, ".jl")
            continue
        elseif startswith(file, "test_")
            include(file_path)
        end
    end
end

@testset verbose = true failfast = true begin
    if length(ARGS) > 0
        include(joinpath(@__DIR__, ARGS[1]))
    else
        recursive_include(@__DIR__)
    end
end

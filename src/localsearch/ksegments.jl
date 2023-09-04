function compute_ksegments_distances(data::AbstractMatrix{<:Real})
    n, d = size(data)

    means = zeros(n, n, d)
    for i in 1:n
        for j in i:n
            if i == j
                means[i, j, :] = data[i, :]
            else
                means[i, j, :] = (means[i, j-1, :] * (j - i) + data[j, :]) / (j - i + 1)
            end
        end
    end

    distances = zeros(n, n)
    for i in 1:n
        for j in i:n
            for l in i:j
                distances[i, j] += sum((data[l, :] - means[i, j, :]) .^ 2)
            end
        end
    end

    return distances, means
end

function regress_ksegments(data::AbstractMatrix{<:Real}, k::Integer)
    n, d = size(data)

    one_seg_dist, one_seg_mean = compute_ksegments_distances(data)

    # Keep a matrix of the total segmentation costs for any p-segmentation of
    # a subsequence data[1:n] where 1<=p<=k and 1<=n<=n. The extra column at
    # the beginning is an effective zero-th row which allows us to index to
    # the case that a (k-1)-segmentation is actually disfavored to the
    # whole-segment average.
    k_seg_dist = zeros(k, n + 1)

    # Also store a pointer structure which will allow reconstruction of the regression which matches.
    # (Without this information, we'd only have the cost of the regression.)
    k_seg_path = zeros(Int, k, n)

    # Initialize the case k=1 directly from the pre-computed distances
    k_seg_dist[1, 2:end] = one_seg_dist[1, :]

    # Any path with only a single segment has a right (non-inclusive) boundary at the zeroth element.
    k_seg_path[1, :] .= 0

    # Then for p segments through p elements, the right boundary for the (p-1) case must obviously be (p-1).
    for i in 1:k
        k_seg_path[Base._sub2ind(size(k_seg_path), i, i)] = i - 1
    end

    # Now go through all remaining subcases 1 < p <= k
    for p in 2:k
        # Update the substructure as successively longer subsequences are considered.
        for n in p:n
            # Enumerate the choices and pick the best one. Encodes the recursion
            # for even the case where j=1 by adding an extra boundary column on the
            # left side of k_seg_dist. The j-1 indexing is then correct without
            # subtracting by one since the real values need a plus one correction.

            choices = k_seg_dist[p-1, 1:n] + one_seg_dist[1:n, n]

            bestval, bestidx = findmin(choices)

            # Store the sub-problem solution. For the path, store where the (p-1) case's right boundary is located.
            k_seg_path[p, n] = bestidx - 1

            # Then remember to offset the distance information due to the boundary (ghost) cells in the first column.
            k_seg_dist[p, n+1] = bestval
        end
    end

    reg = zeros(n, d)

    # Now use the solution information to reconstruct the optimal regression.
    # Fill in each segment reg(i:j) in pieces, starting from the end where the solution is known.
    rhs = size(reg, 1)
    for p in k:-1:1
        # Get the corresponding previous boundary
        lhs = k_seg_path[p, rhs]

        # The pair (lhs,rhs] is now a half-open interval, so set it appropriately

        for i in lhs+1:rhs
            for j in 1:d
                reg[i, j] = one_seg_mean[lhs + 1, rhs, j]
            end
        end
        # reg[lhs+1:rhs, :] .= one_seg_mean[lhs+1, rhs, :]

        # Update the right edge pointer
        rhs = lhs
    end

    return reg
end

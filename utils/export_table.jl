using CSV, DataFrames

function export_table(x_order, feature_list, intercept, theta, N, feature_case, num_feature, num_order, max_point, expdirpath, filename)

    cutoff = []
    point = []
    selected_feature = []

    for j in 1:num_feature
        found = 0
        for p in 1:max_point
            for t in 1:num_order[j]
                if theta[j,t,p] > 0 && found == 0
                    push!(cutoff, x_order[j][t])
                    push!(point, p)
                    push!(selected_feature, feature_list[j])
                    found = found + 1
                end
            end
        end
    end

    df = DataFrame(selected_feature = selected_feature, cutoff = cutoff, intercept = intercept, point = point)
    CSV.write(expdirpath * "N" * string(N) * "_" * feature_case * "_" * filename * ".csv", df)

end

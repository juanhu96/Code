include("../utils/estimate_include.jl")

using Statistics


function main_IGA(dataset, N, feature_case, quantile_list, MAX_ITER, delta, C0, max_point, max_runtime, num_threads, tol_gap, expdirpath, filename)

    # initialization
    z, feature_list, num_feature, num_obs, num_attr, x_order, num_order, v_order = initial(dataset, N, feature_case)
    
    global I_prev = []
    for j = 1:num_feature

        if num_order[j] <= length(quantile_list)
            push!(I_prev, collect(1:num_order[j]-1))
        else
            push!(I_prev, round.(Int, quantile(collect(1:num_order[j]-1), quantile_list)))
        end

    end

    print(I_prev)



    # IGA
    global iter = 0
    while true

        global intercept, theta, summary = estimate_MICP_IGA(z, I_prev, C0, max_point, num_feature, num_obs, num_attr, x_order, num_order, v_order, max_runtime, tol_gap, num_threads)

        global I_next = update_list(I_prev, theta, num_feature, num_order, max_point, delta)

        # convergence
        if I_next == I_prev || iter >= MAX_ITER
            break
        end

        global I_prev = I_next   
        global iter = iter + 1

    end
    
    print(I_next)

    # export
    export_table(x_order, feature_list, intercept, theta, N, feature_case, num_feature, num_order, max_point, expdirpath, filename)
    print(summary)

end




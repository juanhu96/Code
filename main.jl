include("main_include.jl")

### parameters
# N_list : [1000, 5000, 10000, 20000]
# feature_case_list : ["core", "basic", "full"]
# quantile_list : collect(0.1:0.1:1)

N_list = [10000]
feature_case_list = ["full"]
dataset = "FULL"
quantile_list = collect(0.1:0.1:1)


function main(N_list, feature_case_list)


    C0 = 1e-4
    max_point = 5
    integer_relaxation = false
    max_runtime = 28800 # 8hours
    tol_gap = 1e-3 # 1e-4 by default
    num_threads = 10


    for N in N_list
        for feature_case in feature_case_list
            println("Start with N = $N, C0 = $C0, max_point = $max_point, feature_case = $feature_case")

            # solution_summary = estimate_MICP(N, C0, max_point, feature_case, integer_relaxation, max_runtime, tol_gap, num_threads)
            # println(solution_summary)

            # solution_summary = estimate_MICP_noconic(dataset, N, C0, max_point, feature_case, integer_relaxation, max_runtime, tol_gap, num_threads)
            # println(solution_summary)

            solution_summary = estimate_MICP_quartile(dataset, N, C0, max_point, feature_case, quantile_list, integer_relaxation, max_runtime, tol_gap, num_threads)
            println(solution_summary)

            # solution_summary = estimate_MICP_lazyconstr(N, C0, max_point, feature_case, integer_relaxation, max_runtime, tol_gap, num_threads)
            # println(solution_summary)

        end
    end
end

# main(N_list, feature_case_list)



########################################################################################################
########################################################################################################
########################################################################################################


using Statistics

# parameters
function main_IGA()


    C0 = 10
    max_point = 5
    integer_relaxation = false
    max_runtime = 3600
    MAX_ITER = 5
    tol_gap = 1e-2
    num_threads = 10
    delta = 1e-2
    expdirpath = "../Results/"
    filename = "IGA"

    
    dataset = "FULL"
    N = 10000
    quantile_list = collect(0.1:0.1:1)
    feature_case = "basic"


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
    println(I_prev)


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
    export_table(x_order, feature_list, theta, N, feature_case, num_feature, num_order, max_point, expdirpath, filename)


end


main_IGA()



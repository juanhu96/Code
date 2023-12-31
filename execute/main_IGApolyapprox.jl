include("../utils/estimate_include.jl")

# using Statistics
# using Dates

function main_IGApolyapprox(dataset, N, feature_case, MAX_ITER, MAX_RUNTIME, delta, nu, epsilon, C0, 
    max_point, max_runtime, num_threads, tol_gap, expdirpath, filename)

    # initialization
    z, feature_list, num_feature, num_obs, num_attr, x_order, num_order, v_order = initial(dataset, N, feature_case)
    
    # generate list
    global I_prev, I_tilde_prev, n_min_prev, n_max_prev = initial_I(num_feature, num_order, max_point)

    nu_tilde = exp.(-(nu .- 1))
    num_lin = length(nu)

    # IGA
    global iter = 0
    global optimal_feature = []
    t_start = Dates.now()
    while true

        # solve PolyApprox
        global intercept, theta, summary = estimate_MICP_IGApolyapprox(z, nu, nu_tilde, I_prev, C0, max_point, num_feature, num_obs, num_attr, num_lin, 
        x_order, num_order, v_order, epsilon, max_runtime, tol_gap, num_threads)

        # update I and I tilde
        global I_next, I_tilde_next, n_min_next, n_max_next, optimal_feature = update_I(I_prev, I_tilde_prev, n_min_prev, n_max_prev, optimal_feature, theta, num_feature, num_order, max_point, delta)

        time = round(Dates.now() - t_start, Second, RoundUp)

        if iter >= MAX_ITER || time.value >= MAX_RUNTIME || I_tilde_next == I_tilde_prev 
            if iter >= MAX_ITER 
                println("Max iteration achieved")
            elseif time.value >= MAX_RUNTIME
                println("Max runtime achieved")
            else
                println("Converged")
            end
            
            println("Time spent: " * string(time.value) * "; Iteration spent: " * string(iter))

            break   
        end

        global I_prev = I_next   
        global I_tilde_prev = I_tilde_next   
        global n_min_prev = n_min_next   
        global n_max_prev = n_max_next
        global iter = iter + 1

    end

    # export
    print(summary)
    export_table(x_order, feature_list, intercept, theta, N, feature_case, num_feature, num_order, max_point, expdirpath, filename)

end




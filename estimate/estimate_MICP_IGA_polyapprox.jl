using CSV, Dates, DataFrames, LinearAlgebra, JuMP, MosekTools


function estimate_MICP_IGApolyapprox(z, nu, nu_tilde, I_list, C0, max_point, num_feature, num_obs, num_attr, num_lin, 
    x_order, num_order, v_order, epsilon, max_runtime, tol_gap, num_threads)
    

    model = Model(Mosek.Optimizer)
    set_attribute(model, "MSK_IPAR_NUM_THREADS", num_threads)
    set_attribute(model, "QUIET", false)
    set_attribute(model, "MSK_DPAR_MIO_TOL_REL_GAP", tol_gap) # 3%
    set_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", max_runtime) # MSK_DPAR_MIO_MAX_TIME


    # variables 
    theta = @variable(model, theta[j = 1:num_feature, t = 1:num_order[j], p = 1:max_point], lower_bound = 0, upper_bound = 1)
    for j = 1:num_feature
        for t in I_list[j]
            for p = 1:max_point
                set_binary(theta[j,t,p])
            end
        end
    end

    q = @variable(model, q[1:num_feature, 1:max_point], Bin)
    v = @variable(model, v[1:num_feature], Bin)
    u0 = @variable(model, u0, Int)
    u = @variable(model, u[1:num_obs])
    eta = @variable(model, eta[1:num_obs, 1:2])
    phi = @variable(model, phi[1:num_obs])


    # constraints
    cons1 = @constraint(model, [j=1:num_feature, t=1:num_order[j]-1, p=1:max_point], theta[j,t,p] <= theta[j,t+1,p])
    cons2 = @constraint(model, [j=1:num_feature, p=1:max_point], theta[j,num_order[j],p] == q[j,p])
    cons2_0 = @constraint(model, [j=1:num_feature, p=1:max_point], theta[j,1,p] == 0)
    cons3 = @constraint(model, [j=1:num_feature], sum(q[j, p] for p=1:max_point) == v[j])
    cons4 = @constraint(model, [i = 1:num_obs], u[i] == u0 + 
        sum(p * sum(theta[j, v_order[j][i], p] for j=1:num_feature) for p=1:max_point))

    cons5 = @constraint(model, [i = 1:num_obs], eta[i, 1] + eta[i, 2] <= 1)

    # POLY APPROX
    cons6 = @constraint(model, [i = 1:num_obs, l = 1:num_lin], eta[i, 1] * nu_tilde[l] + nu[l] + phi[i] >= -epsilon)
    cons7 = @constraint(model, [i = 1:num_obs, l = 1:num_lin], eta[i, 2] * nu_tilde[l] + nu[l] - (u[i]-phi[i]) >= -epsilon)


    obj = @objective(model, Min, sum(-z[i] * u[i] + phi[i] for i=1:num_obs) + 
        C0 * sum(v[j] for j=1:num_feature))


    optimize!(model)

    if termination_status(model) == OPTIMAL
        println("Solution is optimal")
    elseif termination_status(model) == max_runtime && has_values(model)
        println("Solution is suboptimal due to a time limit, but a primal solution is available")
    else
        println("Something else")
    end
    println("  objective value = ", objective_value(model))
    
    return JuMP.value.(u0), JuMP.value.(theta), solution_summary(model)


end






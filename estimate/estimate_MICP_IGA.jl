using CSV, Dates, DataFrames, LinearAlgebra, JuMP, MosekTools


function estimate_MICP_IGA(z, I_list, C0, max_point, num_feature, num_obs, num_attr, x_order, num_order, v_order, max_runtime, tol_gap, num_threads)
    

    model = Model(Mosek.Optimizer)
    set_attribute(model, "MSK_IPAR_NUM_THREADS", num_threads)
    set_attribute(model, "QUIET", true)
    set_attribute(model, "MSK_DPAR_MIO_TOL_REL_GAP", tol_gap) # 3%
    set_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", max_runtime) # MSK_DPAR_MIO_MAX_TIME


    # variables 
    theta = @variable(model, theta[j = 1:num_feature, t = 1:num_order[j], p = 1:max_point], lower_bound = 0, upper_bound = 1)
    for j = 1:num_feature
        for p = 1:max_point
            for t in I_list[j][p]
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
    cons6 = @constraint(model, [i = 1:num_obs], [-phi[i], 1, eta[i, 1]] in MOI.ExponentialCone())
    cons7 = @constraint(model, [i = 1:num_obs], [u[i]-phi[i], 1, eta[i, 2]] in MOI.ExponentialCone())


    obj = @objective(model, Min, sum(-z[i] * u[i] + phi[i] for i=1:num_obs) + 
        C0 * sum(v[j] for j=1:num_feature))


    optimize!(model)

    
    return JuMP.value.(u0), JuMP.value.(theta), solution_summary(model)


end






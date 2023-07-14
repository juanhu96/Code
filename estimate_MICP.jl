using CSV, Dates, DataFrames, LinearAlgebra, JuMP, MosekTools


function estimate_MICP(N, C0, max_point, feature_case, integer_relaxation, max_runtime, tol_gap, num_threads)


    model = Model(Mosek.Optimizer)
    set_attribute(model, "MSK_IPAR_NUM_THREADS", num_threads)
    set_attribute(model, "QUIET", false)
    set_attribute(model, "MSK_DPAR_MIO_TOL_REL_GAP", tol_gap) # 3%
    set_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", max_runtime) # MSK_DPAR_MIO_MAX_TIME

    # variables
    if integer_relaxation == true
        theta = @variable(model, theta[j = 1:num_feature, t = 1:num_order[j], p = 1:max_point], lower_bound = 0, upper_bound = 1.0)
        q = @variable(model, q[1:num_feature, 1:max_point], lower_bound = 0, upper_bound = 1.0)
        v = @variable(model, v[1:num_feature], lower_bound = 0, upper_bound = 1.0)
    else
        theta = @variable(model, theta[j = 1:num_feature, t = 1:num_order[j], p = 1:max_point], Bin)
        q = @variable(model, q[1:num_feature, 1:max_point], Bin)
        v = @variable(model, v[1:num_feature], Bin)
    end

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

    t = Dates.now() - t_start
    println("The time for constructing optimization model is $t, start solving the model")

    optimize!(model)

    return JuMP.value.(u0), JuMP.value.(theta), solution_summary(model)

end
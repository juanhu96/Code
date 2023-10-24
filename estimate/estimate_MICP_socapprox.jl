using CSV, Dates, DataFrames, LinearAlgebra, JuMP, SCS


function estimate_MICP_socapprox(z, L, C0, max_point, num_feature, num_obs, 
    x_order, num_order, v_order, max_runtime, tol_gap, num_threads)

    model = Model(Mosek.Optimizer)
    # model = Model(SCS.Optimizer)
    
    set_attribute(model, "MSK_IPAR_NUM_THREADS", num_threads)
    set_attribute(model, "QUIET", false)
    set_attribute(model, "MSK_DPAR_MIO_TOL_REL_GAP", tol_gap) # 3%
    set_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", max_runtime) # MSK_DPAR_MIO_MAX_TIME

    # variables 
    theta = @variable(model, theta[j = 1:num_feature, t = 1:num_order[j], p = 1:max_point], Bin)
    q = @variable(model, q[1:num_feature, 1:max_point], Bin)
    v = @variable(model, v[1:num_feature], Bin)


    u0 = @variable(model, -max_point <= u0 <= max_point, Int)
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


    # SOC approx for first cone constraint
    x = @variable(model, x[1:num_obs, 1:2])
    alpha = @variable(model, alpha[1:num_obs, 1:2] >= 0)
    y = @variable(model, y[1:num_obs])
    z = @variable(model, z[1:num_obs])
    f = @variable(model, f[1:num_obs] >= 0)
    g = @variable(model, g[1:num_obs] >= 0)
    h = @variable(model, h[1:num_obs] >= 0)
    nu = @variable(model, nu[1:num_obs, 1:L] >= 0)

    @constraint(model, [i = 1:num_obs], x[i, 1] + x[i, 2] == -phi[i])
    @constraint(model, [i = 1:num_obs], alpha[i, 1] + alpha[i, 2] == 1)
    @constraint(model, [i = 1:num_obs], y[i] == x[i, 2] / 2^L)
    @constraint(model, [i = 1:num_obs], z[i] == alpha[i, 1] +  x[i, 2]/ 2^L)
    @constraint(model, [i = 1:num_obs], (23*alpha[i, 2] + 20*y[i] + 6*f[i] + h[i])/24 <= nu[i, 1]) # nu here is different from polyapprox
    @constraint(model, [i = 1:num_obs], [alpha[i, 2]; f[i]/2; y[i]] in RotatedSecondOrderCone())
    @constraint(model, [i = 1:num_obs], [alpha[i, 2]; g[i]/2; z[i]] in RotatedSecondOrderCone())
    @constraint(model, [i = 1:num_obs], [alpha[i, 2]; h[i]/2; g[i]] in RotatedSecondOrderCone())
    @constraint(model, [i = 1:num_obs, l = 1:L-1], [alpha[i, 2]; nu[i, l+1]/2; nu[i, l]] in RotatedSecondOrderCone())
    @constraint(model, [i = 1:num_obs], [alpha[i, 2]; eta[i, 1]/2; nu[i, L]] in RotatedSecondOrderCone())

    @constraint(model, [i = 1:num_obs], x[i, 1] <= -20 * alpha[i, 1])
    @constraint(model, [i = 1:num_obs], x[i, 2] >= -20 * alpha[i, 2])
    @constraint(model, [i = 1:num_obs], x[i, 2] <= 60 * alpha[i, 2])


    # SOC approx for second cone constraint (_hat)

    x_hat = @variable(model, x_hat[1:num_obs, 1:2])
    alpha_hat = @variable(model, alpha_hat[1:num_obs, 1:2] >= 0)
    y_hat = @variable(model, y_hat[1:num_obs])
    z_hat = @variable(model, z_hat[1:num_obs])
    f_hat = @variable(model, f_hat[1:num_obs] >= 0)
    g_hat = @variable(model, g_hat[1:num_obs] >= 0)
    h_hat = @variable(model, h_hat[1:num_obs] >= 0)
    nu_hat = @variable(model, nu_hat[1:num_obs, 1:L] >= 0)


    @constraint(model, [i = 1:num_obs], x_hat[i, 1] + x_hat[i, 2] == u[i]-phi[i]) # diff 1
    @constraint(model, [i = 1:num_obs], alpha_hat[i, 1] + alpha_hat[i, 2] == 1)
    @constraint(model, [i = 1:num_obs], y_hat[i] == x_hat[i, 2] / 2^L)
    @constraint(model, [i = 1:num_obs], z_hat[i] == alpha_hat[i, 1] +  x_hat[i, 2]/ 2^L)
    @constraint(model, [i = 1:num_obs], (23*alpha_hat[i, 2] + 20*y_hat[i] + 6*f_hat[i] + h_hat[i])/24 <= nu_hat[i, 1])

    @constraint(model, [i = 1:num_obs], [alpha_hat[i, 2]; f_hat[i]/2; y_hat[i]] in RotatedSecondOrderCone())
    @constraint(model, [i = 1:num_obs], [alpha_hat[i, 2]; g_hat[i]/2; z_hat[i]] in RotatedSecondOrderCone())
    @constraint(model, [i = 1:num_obs], [alpha_hat[i, 2]; h_hat[i]/2; g_hat[i]] in RotatedSecondOrderCone())
    @constraint(model, [i = 1:num_obs, l = 1:L-1], [alpha_hat[i, 2]; nu_hat[i, l+1]/2; nu_hat[i, l]] in RotatedSecondOrderCone())
    @constraint(model, [i = 1:num_obs], [alpha_hat[i, 2]; eta[i, 2]/2; nu_hat[i, L]] in RotatedSecondOrderCone())  # diff 2

    @constraint(model, [i = 1:num_obs], x_hat[i, 1] <= -20 * alpha_hat[i, 1])
    @constraint(model, [i = 1:num_obs], x_hat[i, 2] >= -20 * alpha_hat[i, 2])
    @constraint(model, [i = 1:num_obs], x_hat[i, 2] <= 60 * alpha_hat[i, 2])


    # objective
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



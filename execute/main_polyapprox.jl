include("../utils/estimate_include.jl")



# Initial fixed cut
function main_polyapprox(dataset, N, feature_case, nu, epsilon, C0, max_point, max_runtime, num_threads, tol_gap, expdirpath, filename)

    z, feature_list, num_feature, num_obs, num_attr, x_order, num_order, v_order = initial(dataset, N, feature_case)

    nu_tilde = exp.(-(nu .- 1))
    num_lin = length(nu)

    intercept, theta, summary = estimate_MICP_polyapprox(z, nu, nu_tilde, C0, max_point, num_feature, num_obs, num_attr, num_lin, 
    x_order, num_order, v_order, epsilon, max_runtime, tol_gap, num_threads)


    export_table(x_order, feature_list, intercept, theta, N, feature_case, num_feature, num_order, max_point, expdirpath, filename)


end




include("main_include.jl")



function main_polyapprox()

    C0 = 10
    max_point = 5
    integer_relaxation = false
    max_runtime = 7200
    tol_gap = 1e-3 # 1e-4 by default
    num_threads = 10

    
    N = 10000
    feature_case = "full"
    dataset = "SAMPLE"
    quantile_list = collect(0.1:0.1:1)


    z, feature_list, num_feature, num_obs, num_attr, x_order, num_order, v_order = initial(dataset, N, feature_case)


    epsilon = 1e-1
    nu = collect(1:1:10) # linearization
    nu_tilde = exp.(-(nu .- 1))
    num_lin = length(nu)

    print(nu, nu_tilde)

    intercept, theta, summary = estimate_MICP_polyapprox(z, nu, nu_tilde, C0, max_point, num_feature, num_obs, num_attr, num_lin, 
    x_order, num_order, v_order, epsilon, max_runtime, tol_gap, num_threads)

    expdirpath = "../Results/"
    filename = "polyapprox"

    export_table(x_order, feature_list, intercept, theta, N, feature_case, num_feature, num_order, max_point, expdirpath, filename)


end


main_polyapprox()
include("../utils/estimate_include.jl")



function main_original(dataset, N, feature_case, C0, max_point, max_runtime, num_threads, tol_gap, expdirpath, filename)
    
    z, feature_list, num_feature, num_obs, num_attr, x_order, num_order, v_order = initial(dataset, N, feature_case)

    intercept, theta, summary = estimate_MICP(z, C0, max_point, num_feature, num_obs, num_attr,
    x_order, num_order, v_order, max_runtime, tol_gap, num_threads)
    
    print(summary)

    export_table(x_order, feature_list, intercept, theta, N, feature_case, num_feature, num_order, max_point, expdirpath, filename)

end


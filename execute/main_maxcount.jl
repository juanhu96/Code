include("../utils/estimate_include.jl")


function main_maxcount(dataset, N, feature_case, C0, max_point, max_runtime, num_threads, tol_gap, expdirpath, filename)

    z, feature_list, num_feature, num_obs, num_attr, x_order, num_order, v_order = initial(dataset, N, feature_case)

    z_modified = [x == 0 ? -1 : x for x in z] # relabel

    intercept, theta, summary = estimate_MICP_maxcount(z_modified, C0, max_point, num_feature, num_obs, 
    x_order, num_order, v_order, max_runtime, tol_gap, num_threads)

    export_table(x_order, feature_list, intercept, theta, N, feature_case, num_feature, num_order, max_point, expdirpath, filename)

end
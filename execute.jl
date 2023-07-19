include("utils/main_include.jl")


dataset = "SAMPLE"
feature_case = "basic"
expdirpath = "../../Results/"

C0 = 10
max_point = 5
max_runtime = 7200
num_threads = 20
tol_gap = 1e-4


function execute_main(dataset, feature_case, C0, max_point, max_runtime, num_threads, tol_gap, expdirpath)

    N_list = [1000]

    for N in N_list
        
        println("ORIGINAL")
        main_original(dataset, N, feature_case, C0, max_point, max_runtime, num_threads, tol_gap, expdirpath, "original")
        

        # quantile_list = collect(0.1:0.1:1)
        # MAX_ITER = 5
        # delta = 1e-2
        # main_IGA(dataset, N, feature_case, quantile_list, MAX_ITER, delta, C0, max_point, max_runtime, num_threads, tol_gap, expdirpath, "IGA")

        println("QUARTILE")
        quantile_list = collect(0.1:0.1:1)
        main_quartile(dataset, N, feature_case, quantile_list, C0, max_point, max_runtime, num_threads, tol_gap, expdirpath, "qaurtile")

        println("POLYAPPROX")
        nu = collect(1:1:20)
        epsilon = 1e-3
        main_polyapprox(dataset, N, feature_case, nu, epsilon, C0, max_point, max_runtime, num_threads, tol_gap, expdirpath, "polyapprox")
    
    end


end



execute_main(dataset, feature_case, C0, max_point, max_runtime, num_threads, tol_gap, expdirpath)


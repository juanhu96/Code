include("utils/main_include.jl")


# dataset = "SAMPLE"
# feature_case = "full"
dataset = "Framingham"
feature_case = "Framingham"
expdirpath = "/mnt/phd/jihu/opioid_conic/Results/Framingham/"

C0 = 10
max_point = 5
max_runtime = 7200
num_threads = 20
tol_gap = 1e-2


function execute_main(dataset, feature_case, C0, max_point, max_runtime, num_threads, tol_gap, expdirpath)

    N_list = [1000]

    for N in N_list
        
        # println("****************************** ORIGINAL ******************************")
        # main_original(dataset, N, feature_case, C0, max_point, max_runtime, num_threads, tol_gap, expdirpath, "original")
        
        print("****************************** IGA (20%) ******************************\n")
        MAX_ITER = 10
        MAX_RUNTIME = 7200
        main_IGA(dataset, N, feature_case, MAX_ITER, MAX_RUNTIME, C0, max_point, max_runtime, num_threads, 2e-1, expdirpath, "IGA2e1")

        print("****************************** IGA (5%) ******************************\n")
        MAX_ITER = 10
        MAX_RUNTIME = 7200
        main_IGA(dataset, N, feature_case, MAX_ITER, MAX_RUNTIME, C0, max_point, max_runtime, num_threads, 5e-2, expdirpath, "IGA5e2")

        # println("****************************** QUARTILE ******************************\n")
        # quantile_list = collect(0.1:0.1:1)
        # main_quartile(dataset, N, feature_case, quantile_list, C0, max_point, max_runtime, num_threads, tol_gap, expdirpath, "quartile")

        # println("****************************** POLYAPPROX ******************************\n")
        # nu = collect(-20:1:2)
        # epsilon = 1e-3
        # main_polyapprox(dataset, N, feature_case, nu, epsilon, C0, max_point, max_runtime, num_threads, tol_gap, expdirpath, "poly")

        # println("****************************** IGA_POLYAPPROX ******************************\n")
        # MAX_ITER = 10
        # MAX_RUNTIME = 7200
        # delta = 1e-1
        # nu = collect(-20:1:2)
        # epsilon = 1e-3
        # main_IGApolyapprox(dataset, N, feature_case, MAX_ITER, MAX_RUNTIME, delta, nu, epsilon, C0, max_point, max_runtime, num_threads, 5e-2, expdirpath, "IGApoly")
        
    end
    
end


execute_main(dataset, feature_case, C0, max_point, max_runtime, num_threads, tol_gap, expdirpath)


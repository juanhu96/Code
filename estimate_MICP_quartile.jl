using CSV, Dates, DataFrames, LinearAlgebra, JuMP, MosekTools, Statistics


function estimate_MICP_quartile(dataset, N, C0, max_point, feature_case, quantile_list, integer_relaxation, max_runtime, tol_gap, num_threads)
    
    # initial
    if dataset == "SAMPLE"
        df = DataFrame(CSV.File("../Data/SAMPLE_2018_LONGTERM_stratified_$N.csv"))
    elseif dataset == "FULL"
        df = DataFrame(CSV.File("../Data/FULL_2018_LONGTERM_UPTOFIRST.csv"))
    else
        println("Warning: case undefined")
    end


    if feature_case == "core"
        feature_list = ["consecutive_days", "concurrent_MME"]
        df = select(df, [:consecutive_days, :concurrent_MME, :long_term_180])
        x = df[!, [:consecutive_days, :concurrent_MME]]
    
    elseif feature_case == "basic"
        feature_list = ["concurrent_MME", "concurrent_methadone_MME", 
        "consecutive_days", "num_prescribers", "num_pharmacies", "concurrent_benzo"]
        df = select(df, [:concurrent_MME, :concurrent_methadone_MME, :consecutive_days,
            :num_prescribers, :num_pharmacies, :concurrent_benzo, :long_term_180])
        x = df[!, [:concurrent_MME, :concurrent_methadone_MME, :consecutive_days,
        :num_prescribers, :num_pharmacies, :concurrent_benzo]]

    elseif feature_case == "full"
        feature_list = ["concurrent_MME", "concurrent_methadone_MME", 
        "consecutive_days", "num_prescribers", "num_pharmacies", "concurrent_benzo",
        "age", "num_presc", "dose_diff", "MME_diff", "days_diff", 
        "Codeine", "Hydrocodone", "Oxycodone", "Morphine", "HMFO",
        "ever_switch_drug", "ever_switch_payment"]
        df = select(df, [:concurrent_MME, :concurrent_methadone_MME, :consecutive_days,
            :num_prescribers, :num_pharmacies, :concurrent_benzo,
            :age, :num_presc, :dose_diff, :MME_diff, :days_diff,
            :Codeine, :Hydrocodone, :Oxycodone, :Morphine, :HMFO,
            :ever_switch_drug, :ever_switch_payment, :long_term_180])
        x = df[!, [:concurrent_MME, :concurrent_methadone_MME, :consecutive_days,
        :num_prescribers, :num_pharmacies, :concurrent_benzo,
        :age, :num_presc, :dose_diff, :MME_diff, :days_diff,
        :Codeine, :Hydrocodone, :Oxycodone, :Morphine, :HMFO,
        :ever_switch_drug, :ever_switch_payment, :long_term_180]]
    
    else
        println("Warning: case undefined")
    end


    num_feature = length(feature_list)
    num_obs, num_attr = size(df)
    z = df[!, :long_term_180]

    x_min = collect(minimum(eachrow(x)))
    x_max = collect(maximum(eachrow(x)))

    t_start = Dates.now()
    x_order = []
    num_order = []
    v_order = []
    for feature in feature_list

        if length(unique(x[:, feature])) <= length(quantile_list)
            x_order_item = sort(unique(x[:, feature]))
            v_order_item = map(x_i -> findfirst(c_j -> x_i == c_j, x_order_item), x[:, feature])
        else
            x_order_item = quantile(x[:, feature], quantile_list)
            v_order_item = map(x_i -> findfirst(c_j -> x_i <= c_j, x_order_item), x[:, feature])
        end

        push!(x_order, x_order_item)
        push!(num_order, length(x_order_item))
        push!(v_order, v_order_item)
    end
    t = Dates.now() - t_start
    println("The time for constructing x_order and v_order $t")

    ##################################################################
    
    t_start = Dates.now()
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

    ##################################################################
    # export
    cutoff = []
    point = []
    selected_feature = []
    for j in 1:num_feature
        found = 0
        for p in 1:max_point
            for t in 1:num_order[j]
                if JuMP.value.(theta[j,t,p]) > 0 && found == 0
                    push!(cutoff, x_order[j][t])
                    push!(point, p)
                    push!(selected_feature, feature_list[j])
                    found = found + 1
                end
            end
        end
    end
    intercept = JuMP.value.(u0)

    df = DataFrame(selected_feature = selected_feature, cutoff = cutoff, intercept = intercept, point = point)
    if integer_relaxation == true
        CSV.write("../Results/conic_opt_" * string(N) * "_" * feature_case * "_quartile_lp.csv", df)
    else
        CSV.write("../Results/conic_opt_" * string(N) * "_" * feature_case * "_quartile.csv", df)
    end

    return solution_summary(model)






























end
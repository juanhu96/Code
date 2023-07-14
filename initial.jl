using CSV, Dates, DataFrames, LinearAlgebra


function initial(dataset, N, feature_case)

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
        x_order_item = sort(unique(x[:, feature]))
        v_order_item = map(x_i -> findfirst(c_j -> x_i == c_j, x_order_item), x[:, feature])
        push!(x_order, x_order_item)
        push!(num_order, length(x_order_item))
        push!(v_order, v_order_item)
    end
    t = Dates.now() - t_start
    println("The time for constructing x_order and v_order $t")


    return z, feature_list, num_feature, num_obs, num_attr, x_order, num_order, v_order

end
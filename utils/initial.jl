using CSV, Dates, DataFrames, LinearAlgebra


workdir = "/mnt/phd/jihu/opioid_conic/"


function initial(dataset, N, feature_case)

    if dataset == "SAMPLE"     
        df = DataFrame(CSV.File(workdir * "Data/SAMPLE_2018_LONGTERM_stratified_$N.csv"))

        df = dropmissing!(df, disallowmissing=true)
        filter!(row -> all(x -> x != "NA", row), df)
        for col in names(df)
            df[!, col] = parse.(Float64, df[!, col])
        end

        z = df[!, :long_term_180]

    elseif dataset == "FULL"
        df = DataFrame(CSV.File(workdir * "Data/FULL_2018_LONGTERM_UPTOFIRST.csv"))

        df = dropmissing!(df, disallowmissing=true)
        filter!(row -> all(x -> x != "NA", row), df)
        for col in names(df)
            df[!, col] = parse.(Float64, df[!, col])
        end

        z = df[!, :long_term_180]

    elseif dataset == "Framingham"
        df = DataFrame(CSV.File(workdir * "Data/framingham.csv"))
        
        # drop missing, drop 'NA', convert numerics
        df = dropmissing!(df, disallowmissing=true)
        filter!(row -> all(x -> x != "NA", row), df)
        for c ∈ names(df, String3)
            df[!, c]= parse.(Float64, df[!, c])
        end

        z = df[!, :TenYearCHD]

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
            :Medicaid, :CommercialIns, :Medicare, :CashCredit, :MilitaryIns,
            :WorkersComp, :Other, :IndianNation, :switch_drug, :switch_payment,
            :ever_switch_drug, :ever_switch_payment, :long_term_180])
        x = df[!, [:concurrent_MME, :concurrent_methadone_MME, :consecutive_days,
        :num_prescribers, :num_pharmacies, :concurrent_benzo,
        :age, :num_presc, :dose_diff, :MME_diff, :days_diff,
        :Codeine, :Hydrocodone, :Oxycodone, :Morphine, :HMFO,
        :Medicaid, :CommercialIns, :Medicare, :CashCredit, :MilitaryIns,
        :WorkersComp, :Other, :IndianNation, :switch_drug, :switch_payment,
        :ever_switch_drug, :ever_switch_payment]]

    elseif feature_case == "Framingham"

        feature_list = ["male", "age", "education", "currentSmoker", "cigsPerDay",
        "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "totChol", 
        "sysBP", "diaBP", "BMI", "heartRate", "glucose", "TenYearCHD"]
        df = df[!, Symbol.(feature_list)]        
        x = select!(df, Not([:TenYearCHD]))
        deleteat!(feature_list, findall(x->x=="TenYearCHD",feature_list))

    else
        println("Warning: case undefined")
    end


    num_feature = length(feature_list)
    num_obs, num_attr = size(df)
    

    x_min = collect(minimum(eachrow(x)))
    x_max = collect(maximum(eachrow(x)))


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

    return z, feature_list, num_feature, num_obs, num_attr, x_order, num_order, v_order

end






function initial_quartile(dataset, N, feature_case, quantile_list)
    
    if dataset == "SAMPLE"
        df = DataFrame(CSV.File(workdir * "Data/SAMPLE_2018_LONGTERM_stratified_$N.csv"))
    elseif dataset == "FULL"
        df = DataFrame(CSV.File(workdir * "Data/FULL_2018_LONGTERM_UPTOFIRST.csv"))
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
            :Medicaid, :CommercialIns, :Medicare, :CashCredit, :MilitaryIns,
            :WorkersComp, :Other, :IndianNation, :switch_drug, :switch_payment,
            :ever_switch_drug, :ever_switch_payment, :long_term_180])
        x = df[!, [:concurrent_MME, :concurrent_methadone_MME, :consecutive_days,
        :num_prescribers, :num_pharmacies, :concurrent_benzo,
        :age, :num_presc, :dose_diff, :MME_diff, :days_diff,
        :Codeine, :Hydrocodone, :Oxycodone, :Morphine, :HMFO,
        :Medicaid, :CommercialIns, :Medicare, :CashCredit, :MilitaryIns,
        :WorkersComp, :Other, :IndianNation, :switch_drug, :switch_payment,
        :ever_switch_drug, :ever_switch_payment, :long_term_180]]

    else
        println("Warning: case undefined")
    end
    

    num_feature = length(feature_list)
    num_obs, num_attr = size(df)
    z = df[!, :long_term_180]

    x_min = collect(minimum(eachrow(x)))
    x_max = collect(maximum(eachrow(x)))

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

    return z, feature_list, num_feature, num_obs, num_attr, x_order, num_order, v_order

end


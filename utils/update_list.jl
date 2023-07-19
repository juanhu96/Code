using LinearAlgebra 

function update_list(I_prev, theta, num_feature, num_order, max_point, delta)
    
    I_temp = deepcopy(I_prev)
    
    # add new elements
    for j in 1:num_feature
    
        for t in deleteat!([1:num_order[j]-1;], I_temp[j])  #for t in complement of I_list[j]

            count = 0
        
            for p in 1:max_point
                if theta[j,t,p] >= delta && theta[j,t,p] < 1 && count == 0 # if any p of the j,t violates, add all p of j,t pair
                    push!(I_temp[j], t)
                    count = count + 1
                end
            end
        end
    end


    # sort
    for j in 1:num_feature
        I_temp[j] = sort(I_temp[j])
    end

    return I_temp

end
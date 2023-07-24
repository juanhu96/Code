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



##############################################################################################################
##############################################################################################################
##############################################################################################################



function initial_I(num_feature, num_order, max_point)

    I = []
    I_tilde = []
    n_min = []
    n_max = []

    for j = 1:num_feature
        
        local I_j = []
        local I_tilde_j = []
        local n_min_j = []
        local n_max_j = []
        
        for p = 1:max_point

            push!(n_min_j, 1)
            push!(n_max_j, num_order[j])

            n = floor((1 + num_order[j]) / 2)

            push!(I_j, [n, n+1])
            push!(I_tilde_j, [n, n+1])
        
        end

        push!(I, I_j)
        push!(I_tilde, I_tilde_j)
        push!(n_min, n_min_j)
        push!(n_max, n_max_j)

    end

    return I, I_tilde, n_min, n_max

end




##############################################################################################################
##############################################################################################################
##############################################################################################################



function update_I(I_prev, I_tilde_prev, n_min_prev, n_max_prev, optimal_feature, theta, num_feature, num_order, max_point, delta)
    
    epsilon = 1e-5 # case where theta = 1.588151832265794e-11, or theta = 0.9999999999439342

    I = deepcopy(I_prev)
    I_tilde = deepcopy(I_tilde_prev)
    n_min = deepcopy(n_min_prev)
    n_max = deepcopy(n_max_prev)

    for j in 1:num_feature, p = 1:max_point

        if !(j in optimal_feature) # feature j hasn't reach optimality

            index_one = I_tilde[j][p][1] # n_k
            index_two = I_tilde[j][p][2] # n_k+1

            theta_one = theta[j, index_one, p]
            theta_two = theta[j, index_two, p]

            if theta_one <= epsilon && theta_two <= epsilon && index_two < n_max[j][p] # avoid n_max, n_max + 1
                        
                # update n
                n_min[j][p] = index_two
                n = floor((n_min[j][p] + n_max[j][p]) / 2)

                # update I tilde and n_k+1
                I_tilde[j][p] = [n, n+1]

                # update I
                push!(I[j][p], n)
                push!(I[j][p], n+1)   

            elseif theta_one >= 1-epsilon && theta_two >= 1-epsilon
                        
                    n_max[j][p] = index_one
                    n = floor((n_min[j][p] + n_max[j][p]) / 2)

                    I_tilde[j][p] = [n, n+1]

                    push!(I[j][p], n)
                    push!(I[j][p], n+1)

            elseif theta_one <= epsilon && theta_two >= 1-epsilon
                println("Optimal found for feature " * string(j) * " with point " * string(p))
                push!(optimal_feature, j)
            end    
        
        end

    end

    return I, I_tilde, n_min, n_max, optimal_feature

end




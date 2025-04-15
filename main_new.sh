#!/bin/bash
export OMP_NUM_THREADS=10

### CREATE STUMPS
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2018 LTOUR > output/create_stumps_2018.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2018 Explore all > output/create_stumps_2018_explore.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2019 Explore all > output/create_stumps_2019_explore.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2018 Explore all > output/create_stumps_2018_explore_median.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2019 Explore all > output/create_stumps_2019_explore_median.txt &

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2018 Explore first > output/create_stumps_2018_first_explore.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2018 Explore upto180 > output/create_stumps_2018_upto180_explore.txt &

# ================================================================nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py single 5 6 1e-15 median feature_exactlyonecutoff_group essential1 > output/1e15_median/p5f6_exactlyonecutoff_group.txt &====
### EXPLORE

### UNCONSTRAINED: NUMBER OF CONDITIONS

run_first_set() {
    for i in 6 8; do
        for feature_type in "atmostonecutoff_feature" "atmostonecutoff_group"; do
            nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py single 5 $i 1e-15 median feature_$feature_type essential1 > output/1e15_median/p5f${i}_feature_${feature_type}.txt &
        done
    done
}

run_second_set() {
    for i in 6 8; do
        for feature_type in "exactlyonecutoff_group" "atmosttwocutoff_group"; do
            nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py single 5 $i 1e-15 median feature_$feature_type essential1 > output/1e15_median/p5f${i}_feature_${feature_type}.txt &
        done
    done
}

# infeasible
run_third_set() {
    for i in 6 8; do
        for feature_type in "exactlytwocutoff_group"; do
            nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py single 5 $i 1e-15 median feature_$feature_type essential2 > output/1e15_median/p5f${i}_feature_${feature_type}.txt &
        done
    done
}

run_fourth_set() {
    for i in 6; do
        for feature_type in "atmostonecutoff_feature" "atmostonecutoff_group"; do
            nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py single 5 $i 1e-15 median feature_$feature_type essential1 nodrug > output/1e15_median/p5f${i}_feature_${feature_type}_nodrug.txt &
        done
    done
}

run_fifth_set() {
    for i in 8; do
        for feature_type in "atmostonecutoff_feature" "atmostonecutoff_group"; do
            nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py single 5 $i 1e-15 median feature_$feature_type essential1 nodrug > output/1e15_median/p5f${i}_feature_${feature_type}_nodrug.txt &
        done
    done
}


run_six_set() {
    for i in 8; do
        for feature_type in "atmostonecutoff_feature"; do
            nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py single 5 $i 1e-15 median feature_$feature_type essential8 nodrug > output/1e15_median/p5f${i}_feature_${feature_type}_nodrug.txt &
        done
    done
}


run_seven_set() {
    for i in 8; do
        for feature_type in "atleast_total"; do
            nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py single 5 $i 1e-15 median feature_$feature_type essential4 > output/1e15_median/p5f${i}_feature_${feature_type}_ess4.txt &
            nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py single 5 $i 1e-15 median feature_$feature_type essential3 > output/1e15_median/p5f${i}_feature_${feature_type}_ess3.txt &
        done
    done
}


# ====================================================================
### BASELINE
# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py DecisionTree > output/baseline/DT.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py RandomForest > output/baseline/RF.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py L1 > output/baseline/L1.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py L2 > output/baseline/L2.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py LinearSVM > output/baseline/SVM.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py XGB > output/baseline/XGB.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py NN > output/baseline/NN.txt & 


# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py DecisionTree median > output/baseline/median/DT_median.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py RandomForest median > output/baseline/median/RF_median.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py L1 median > output/baseline/median/L1_median.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py L2 median > output/baseline/median/L2_median.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py LinearSVM median > output/baseline/median/SVM_median.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py XGB median > output/baseline/median/XGB_median.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py NN median > output/baseline/median/NN_median.txt & 

# ====================================================================

# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py DecisionTree median > output/baseline/median/DT_median_Aug25.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py single 5 20 1e-15 median feature_random essential4 > output/baseline/median/LTOUR_test_Aug25.txt & 

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py single 5 6 1e-15 median feature_random essential1 nodrug > output/baseline/median/LTOUR_longacting_Sep26_test.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py single 5 6 1e-15 median feature_7 essential1 nodrug > output/baseline/median/LTOUR_payment_Sep24.txt &

# ====================================================================

nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py single 5 6 1e-15 median feature_all essential1 nodrug > output/stretch_test.txt &
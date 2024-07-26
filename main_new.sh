# case, scenario, max_points, max_features, weight, intercept


### CREATE STUMPS
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2018 LTOUR > output/create_stumps_2018.txt & 

### TRAIN
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 train single 5 10 original 1e-15 LTOUR > output/LTOUR_train_2018_15.txt & 


# ====================================================================
### EXPLORE

### CREATE STUMPS
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2018 Explore all > output/create_stumps_2018_explore.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2018 Explore first > output/create_stumps_2018_first_explore.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2018 Explore upto180 > output/create_stumps_2018_upto180_explore.txt &

### CREATE TABLES
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 train single 5 20 original 1e-15 Explore feature7 essential1 > output/original_1e15_atleast1_new/2018_train_combined_onepergroup.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 train single 5 20 original 1e-15 Explore feature7 essential1 > output/original_1e15_atleast1_new/2018_train_combined_exactlyonepergroup.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 train single 5 20 original 1e-15 Explore feature7 essential1 first > output/original_1e15_atleast1_new/2018_first_train_combined_onepergroup.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 train single 5 20 original 1e-15 Explore feature7 essential1 upto180 > output/original_1e15_atleast1_new/2018_upto180_train_combined_onepergroup.txt & 

### TEST
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 test all tableall > output/original_1e15_atleast1_new/2018_test.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 test first tablefirst > output/original_1e15_atleast1_new/2018_test_first.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 test upto180 tableupto180 > output/original_1e15_atleast1_new/2018_test_upto180.txt & 

### CONIC
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 test conic_one tableconic_one > output/2018_sample_test_conic_one.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 test conic_two tableconic_two > output/2018_sample_test_conic_two.txt & 



# ====================================================================
### BASELINE
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 base_train DecisionTree > output/baseline/2018_DT.txt & 
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 base_train RandomForest > output/baseline/2018_RF.txt & 
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 base_train L1 > output/baseline/2018_L1.txt & 
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 base_train L2 > output/baseline/2018_L2.txt & 
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 base_train SVM > output/baseline/2018_SVM.txt & 
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 base_train XGB > output/baseline/2018_XGB.txt & 
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py 2018 base_train NN > output/baseline/2018_NN.txt & 



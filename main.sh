# case, scenario, max_points, max_features, weight, intercept

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-5 5 > output/train_10_1e-5.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-8 8 > output/train_10_1e-8.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-10 10 > output/train_10_1e-10.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-15 15 > output/train_10_1e-15.txt &

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-8 sink > output/train_10_sink_1e-8.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-10 sink > output/train_10_sink_1e-10.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-15 sink > output/train_10_sink_1e-15.txt &

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-8 num_feature > output/num_feature_1e-8.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-10 num_feature > output/num_feature_1e-10.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-15 num_feature > output/num_feature_1e-15.txt &

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train cv 5 10 original 1 cv > output/cv_two_cutoffs.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test single 5 20 original 1 cv > output/test.txt &

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-8 1 > output/pos_one_cutoff_1e-8.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-10 1 > output/pos_one_cutoff_1e-10.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-15 1 > output/pos_one_cutoff_1e-15.txt &

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test single 5 10 original 1 5 > output/test.txt
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py base_train single 5 10 original 1 5 > output/base_train_test.txt

# ==========================================

# baseline and LTOUR (without average days)
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py base_train single 5 10 original 1 noavgdays > output/base_train_noavgdays.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-08 noavgdays > output/LTOUR_train_noavgdays.txt & 

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test single 5 10 original 1e-08 noavgdays > output/LTOUR_test_noavgdays.txt

# ==========================================
# create stumps (only the first argv matter, the other are dummies) with HPI
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py create_stumps single 5 10 original 1 5 > output/create_stumps_HPI.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-08 HPI > output/LTOUR_train_HPI.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-08 HPI_noavgdays > output/LTOUR_train_HPI_noavgdays.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test single 5 10 original 1e-08 HPI_noavgdays > output/LTOUR_test_HPI_noavgdays.txt & 

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original 1e-15 HPI_noavgdays_must > output/LTOUR_train_HPI_noavgdays_must.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test single 5 10 original 1e-15 HPI_noavgdays_must > output/LTOUR_test_HPI_noavgdays_must.txt & 



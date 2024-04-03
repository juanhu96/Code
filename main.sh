# case, scenario, max_points, max_features, weight, intercept

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 6 original flexible > output/train_6.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original flexible > output/train_10.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 20 original flexible > output/train_20.txt &

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test single 5 6 original > output/test_6.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test single 5 10 original > output/test_10.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test single 5 20 original > output/test_20.txt &


# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test single 5 10 original firstround > output/test_10_firstround.txt
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test single 5 10 original secondround > output/test_10_secondround.txt

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


nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py base_train single 5 10 original 1 5 > output/base_train_test.txt
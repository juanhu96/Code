# case, scenario, max_points, max_features, weight, intercept

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 6 original flexible > output/train_6.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original flexible > output/train_10.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 20 original flexible > output/train_20.txt &

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test single 5 6 original > output/test_6.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test single 5 10 original > output/test_10.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test single 5 20 original > output/test_20.txt &


nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 10 original firstround > output/train_10_firstround.txt &
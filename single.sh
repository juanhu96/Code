nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 20 original 1e-8 20 > output/20_feature_1e-8.txt &
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 20 original 1e-10 20 > output/20_feature_1e-10.txt &
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 20 original 1e-15 20 > output/20_feature_1e-15.txt &



nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train cv 5 10 original 1 cv_neg_two > output/cv_neg_two.txt &
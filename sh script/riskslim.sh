export OMP_NUM_THREADS=10
mkdir -p ../output/riskslim
mkdir -p ../output/riskslim/percentile

# All presc (f6_nozip_exact)
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub15 interceptlb5 feature_nopatientzip cutoff_atmostone_group essential1 > ../output/riskslim/percentile/LTOUR_p3_f6_nozip.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures7 c01e-15 interceptub15 interceptlb5 feature_nopatientzip cutoff_atmostone_group essential1 > ../output/riskslim/percentile/LTOUR_p3_f7_nozip.txt &

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub15 interceptlb5 feature_nopatientzip cutoff_atmostone_group essential1 exact > ../output/riskslim/percentile/LTOUR_p3_f6_nozip_exact.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures7 c01e-15 interceptub15 interceptlb5 feature_nopatientzip cutoff_atmostone_group essential1 exact > ../output/riskslim/percentile/LTOUR_p3_f7_nozip_exact.txt &

# Naive presc (f6_nozip_exact)
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub15 interceptlb5 feature_nopatientzip cutoff_atmostone_group essential1 first > ../output/riskslim/percentile/LTOUR_p3_f6_nozip_naive.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures7 c01e-15 interceptub15 interceptlb5 feature_nopatientzip cutoff_atmostone_group essential1 first > ../output/riskslim/percentile/LTOUR_p3_f7_nozip_naive.txt &

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub15 interceptlb5 feature_nopatientzip cutoff_atmostone_group essential1 exact first > ../output/riskslim/percentile/LTOUR_p3_f6_nozip_exact_naive.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures7 c01e-15 interceptub15 interceptlb5 feature_nopatientzip cutoff_atmostone_group essential1 exact first > ../output/riskslim/percentile/LTOUR_p3_f7_nozip_exact_naive.txt &


# Test
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test tableLTOUR_6 > ../output/riskslim/percentile/LTOUR_6_test_sep12.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test first tableLTOUR_6 > ../output/riskslim/percentile/LTOUR_6_test_first_sep12.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test first tableLTOUR_naive_6 > ../output/riskslim/percentile/LTOUR_naive_6_test.txt &

# Gender-specific models
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub15 interceptlb5 feature_nopatientzip cutoff_atmostone_group essential1 exact gender_male > ../output/riskslim/percentile/LTOUR_p3_f6_nozip_exact_male.txt &
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub15 interceptlb5 feature_nopatientzip cutoff_atmostone_group essential1 exact gender_female > ../output/riskslim/percentile/LTOUR_p3_f6_nozip_exact_female.txt &
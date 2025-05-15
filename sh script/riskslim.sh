export OMP_NUM_THREADS=10
mkdir -p ../output/riskslim

# SINGLE TRAIN
#!/bin/bash

# for maxpoint in 2 3; do
#   for intercept in 7 8 9 10; do
#     intercept_lb=$((intercept - 1))
#     out="../output/riskslim/LTOUR_p${maxpoint}_f6_c01e-15_intercept${intercept}_median_all_atmostone.txt"
#     nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint${maxpoint} maxfeatures6 c01e-15 interceptub${intercept} interceptlb${intercept_lb} median feature_all cutoff_atmostone_group essential1 nodrug > "$out" &
#     echo "Started: maxpoint${maxpoint}, intercept${intercept} to $out"
#   done
# done


# for maxpoint in 2 3; do
#   for intercept in 7 8 9 10; do
#     intercept_lb=$((intercept - 1))
#     out="../output/riskslim/LTOUR_p${maxpoint}_f6_c01e-15_intercept${intercept}_median_nodemo_atmostone.txt"
#     nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint${maxpoint} maxfeatures6 c01e-15 interceptub${intercept} interceptlb${intercept_lb} median feature_nodemo cutoff_atmostone_group essential1 nodrug > "$out" &
#     echo "Started: maxpoint${maxpoint}, intercept${intercept} → $out"
#   done
# done


# FINAL LTOUR

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug > ../output/riskslim/LTOUR_final.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug > ../output/riskslim/LTOUR_final_days_supply.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug > ../output/riskslim/LTOUR_final_days_supply_no21.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug > ../output/riskslim/noMME/LTOUR_final_f6.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures7 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug > ../output/riskslim/noMME/LTOUR_final_f7.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures8 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug > ../output/riskslim/noMME/LTOUR_final_f8.txt &

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub10 interceptlb5 median feature_6 cutoff_atmostone_group essential1 nodrug > ../output/riskslim/noMME/LTOUR_final_f6_feature6.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures7 c01e-15 interceptub10 interceptlb5 median feature_6 cutoff_atmostone_group essential1 nodrug > ../output/riskslim/noMME/LTOUR_final_f7_feature6.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures8 c01e-15 interceptub10 interceptlb5 median feature_6 cutoff_atmostone_group essential1 nodrug > ../output/riskslim/noMME/LTOUR_final_f8_feature6.txt &


# FIRST PRESC
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug first > ../output/riskslim/noMME/LTOUR_first_f6.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures7 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug first > ../output/riskslim/noMME/LTOUR_first_f7.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures8 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug first > ../output/riskslim/noMME/LTOUR_first_f8.txt &

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub10 interceptlb5 median feature_6 cutoff_atmostone_group essential1 nodrug first > ../output/riskslim/noMME/LTOUR_first_f6_feature6.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures7 c01e-15 interceptub10 interceptlb5 median feature_6 cutoff_atmostone_group essential1 nodrug first > ../output/riskslim/noMME/LTOUR_first_f7_feature6.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures8 c01e-15 interceptub10 interceptlb5 median feature_6 cutoff_atmostone_group essential1 nodrug first > ../output/riskslim/noMME/LTOUR_first_f8_feature6.txt &


# drop days supply (running)
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 > ../output/riskslim/nodays/LTOUR_f6.txt &
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures7 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 exact > ../output/riskslim/nodays/LTOUR_f7_exact.txt &
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures8 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 exact > ../output/riskslim/nodays/LTOUR_f8_exact.txt &

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub10 interceptlb5 median feature_6 cutoff_atmostone_group essential1 > ../output/riskslim/nodays/LTOUR_f6_feature6.txt &
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures7 c01e-15 interceptub10 interceptlb5 median feature_6 cutoff_atmostone_group essential1 exact > ../output/riskslim/nodays/LTOUR_f7_feature6_exact.txt &
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures8 c01e-15 interceptub10 interceptlb5 median feature_6 cutoff_atmostone_group essential1 exact > ../output/riskslim/nodays/LTOUR_f8_feature6_exact.txt &


# keep days supply, set conditions to 7 & 8 (running)
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures7 c01e-15 interceptub15 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug exact > ../output/riskslim/noMME/LTOUR_f7_exact.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures8 c01e-15 interceptub15 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug exact > ../output/riskslim/noMME/LTOUR_f8_exact.txt &

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures7 c01e-15 interceptub15 interceptlb5 median feature_6 cutoff_atmostone_group essential1 nodrug exact > ../output/riskslim/noMME/LTOUR_f7_feature6_exact.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures8 c01e-15 interceptub15 interceptlb5 median feature_6 cutoff_atmostone_group essential1 nodrug exact > ../output/riskslim/noMME/LTOUR_f8_feature6_exact.txt &


# TEST
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test median > ../output/riskslim/LTOUR_test.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test median > ../output/riskslim/LTOUR_test_county.txt & 

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test median > ../output/riskslim/LTOUR_days_supply_test.txt & 
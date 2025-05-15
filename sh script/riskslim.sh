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
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug > ../output/riskslim/noMME/LTOUR_final_f6.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures7 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug > ../output/riskslim/noMME/LTOUR_final_f7.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures8 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug > ../output/riskslim/noMME/LTOUR_final_f8.txt &

# FIRST PRESC
# for maxfeatures in 6 7; do
#     nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures${maxfeatures} c01e-15 interceptub15 interceptlb5 median feature_6 cutoff_atmostone_group essential1 first exact > ../output/riskslim/noMME/LTOUR_first_f${maxfeatures}_feature6_exact.txt &
#     # no patient zip
#     nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures${maxfeatures} c01e-15 interceptub15 interceptlb5 median feature_nopatientzip cutoff_atmostone_group essential1 first exact > ../output/riskslim/noMME/LTOUR_first_f${maxfeatures}_featurenozip_exact.txt &
# done


# keep days supply, set conditions to 7 & 8 (running)
# for maxfeatures in 6 7 8; do
#     nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures${maxfeatures} c01e-15 interceptub15 interceptlb5 median feature_6 cutoff_atmostone_group essential1 exact > ../output/riskslim/noMME/LTOUR_f${maxfeatures}_feature6_exact.txt &
#     # for daily dose
#     nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures${maxfeatures} c01e-15 interceptub15 interceptlb5 median feature_6 cutoff_atmostone_groupdose essential1 exact > ../output/riskslim/noMME/LTOUR_f${maxfeatures}_feature6_exact_dailydose.txt &
# done
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures8 c01e-15 interceptub15 interceptlb5 median feature_nopatientzip cutoff_atmostone_group essential1 exact > ../output/riskslim/noMME/LTOUR_f8_featurenozip_exact.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures8 c01e-15 interceptub15 interceptlb5 median feature_nopatientzip cutoff_atmostone_groupdose essential1 exact > ../output/riskslim/noMME/LTOUR_f8_featurenozip_exact_dailydose.txt &


# at least one MME
nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures7 c01e-15 interceptub15 interceptlb5 median feature_mme cutoff_atmostone_groupMME essential1 exact > ../output/riskslim/MME/LTOUR_f7_featureMME_exact.txt &



# ============================== TEST ==============================

# daily dose from 50 to 100
# for val in 50 60 70 80 90 100; do
#     nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test median tableLTOUR_seven_${val} > ../output/riskslim/noMME/LTOUR_test_seven_${val}.txt &
# done


# compare full table with naive table for first presc
# for table in tableLTOUR_seven_100 tableLTOUR_naive_6 tableLTOUR_naive_7; do
#     out_name=$(echo "$table" | sed 's/tableLTOUR_//')
#     nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test median "$table" first > "../output/riskslim/noMME/LTOUR_test_${out_name}_first.txt" &
# done



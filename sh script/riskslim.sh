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

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug > ../output/riskslim/LTOUR_final.txt &


# TEST
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test median > ../output/riskslim/LTOUR_test.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test median > ../output/riskslim/LTOUR_test_county.txt & 

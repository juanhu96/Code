export OMP_NUM_THREADS=10
mkdir -p ../output/riskslim/county

# - Underestimate: Kern, San Bernardino, Riverside, Fresno
# - Overestimate: Los Angeles, San Francisco, San Mateo

# counties=("Los Angeles" "San Francisco" "San Mateo" "Kern" "San Bernardino" "Riverside" "Fresno")
# for county in "${counties[@]}"; do
#   nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub10 interceptlb5 median feature_all cutoff_atmostone_group essential1 nodrug county"$county" > "../output/riskslim/county/LTOUR_$county.txt" &
# done


# counties=("San Francisco" "Kern")
# for county in "${counties[@]}"; do
#   nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test median county"$county" > "../output/riskslim/county/Test_$county.txt" &
# done


python3 /mnt/phd/jihu/opioid/Code/src/county_summary.py 2018 &
python3 /mnt/phd/jihu/opioid/Code/src/county_summary.py 2019 &
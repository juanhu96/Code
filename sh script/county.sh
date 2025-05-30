export OMP_NUM_THREADS=10
mkdir -p ../output/riskslim/county

### CREATE COUNTY TABLE
# - Underestimate: Kern, San Bernardino, Riverside, Fresno
# - Overestimate: Los Angeles, San Francisco, San Mateo

# counties=("Los Angeles" "San Francisco" "San Mateo" "Kern" "San Bernardino" "Riverside" "Fresno" "Modoc")
# counties=("Humboldt" "Imperial" "Mendocino" "Napa" "San Benito")
# for county in "${counties[@]}"; do
#   nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures7 c01e-15 interceptub15 interceptlb5 median feature_nopatientzip cutoff_atmostone_group essential1 exact county"$county" > "../output/riskslim/county/LTOUR_$county.txt" &
# done

### LTOUR VS. COUNTY TABLE
# counties=("San Francisco" "Kern")
# for county in "${counties[@]}"; do
#   nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test median county"$county" > "../output/riskslim/county/Test_$county.txt" &
# done

### COUNTY SUMMARY
# counties=("Fresno" "San Bernardino" "Los Angeles" "Humboldt" "San Benito" "Riverside")
# for county in "${counties[@]}"; do
#   nohup python3 /mnt/phd/jihu/opioid/Code/src/utils/summary_stats.py total county"$county" > "../output/riskslim/county/summary_stats_$county.txt" &
# done
# python3 /mnt/phd/jihu/opioid/Code/src/utils/merge_tables.py

### COUNTY SUMMARY FOR REASONING BEHIND THE DIFFERENCE IN LTOUR AND COUNTY TABLE
# nohup python3 /mnt/phd/jihu/opioid/Code/src/utils/county_summary.py 2018 > ../output/riskslim/county/condition_stats_2018.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/src/utils/county_summary.py 2019 > ../output/riskslim/county/condition_stats_2019.txt &

### LTOUR BY COUNTY
counties=("Fresno" "San Bernardino" "Los Angeles" "Humboldt" "San Benito" "Riverside")
for county in "${counties[@]}"; do
#   nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test median tableLTOUR county"$county" > "../output/riskslim/county/Test_LTOUR_$county.txt" &
  nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test median first tableLTOUR county"$county" > "../output/riskslim/county/Test_LTOUR_naive_$county.txt" &
done
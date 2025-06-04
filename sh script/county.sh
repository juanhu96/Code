export OMP_NUM_THREADS=10
mkdir -p ../output/riskslim/county

### CREATE COUNTY TABLE
# - Underestimate: Kern, San Bernardino, Riverside, Fresno
# - Overestimate: Los Angeles, San Francisco, San Mateo

# counties=("Los Angeles" "San Francisco" "San Mateo" "Kern" "San Bernardino" "Riverside" "Fresno" "Modoc")
# counties=("Humboldt" "Imperial" "Mendocino" "Napa" "San Benito")
# counties=("Orange" "San Diego" "Alameda" "Santa Clara")
# counties=("Butte" "Shasta" "Tulare" "Stanislaus" "Merced" "San Luis Obispo" "San Joaquin")
# counties=("Sacramento" "Solano" "Sonoma" "Ventura" "Contra Costa" "Monterey")
# counties=("Los Angeles" "San Francisco" "San Mateo" "Kern" "San Bernardino" "Riverside" "Fresno" "Butte" "Stanislaus" "Tulare" "Santa Clara" "Orange" "San Diego" "Alameda")
# for county in "${counties[@]}"; do
#   nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single maxpoint3 maxfeatures6 c01e-15 interceptub15 interceptlb5 feature_nopatientzip cutoff_atmostone_group essential1 exact county"$county" > "../output/riskslim/county/LTOUR_$county.txt" &
# done

### LTOUR VS. COUNTY TABLE
# counties=("San Francisco" "Kern")
# for county in "${counties[@]}"; do
#   nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test median county"$county" > "../output/riskslim/county/Test_$county.txt" &
# done

### COUNTY SUMMARY
# counties=("Fresno" "San Bernardino" "Los Angeles" "Humboldt" "San Benito" "Riverside")
# Underestimate: San Bernardino / Modoc / Kern
# Overestimate: Los Angeles / San Francisco / Santa Clara
# counties=("Kern" "Los Angeles" "Modoc" "San Bernardino" " San Francisco" "Santa Clara")
# for county in "${counties[@]}"; do
#   nohup python3 /mnt/phd/jihu/opioid/Code/src/utils/summary_stats.py total county"$county" > "../output/riskslim/county/summary_stats_$county.txt" &
# done
# nohup python3 /mnt/phd/jihu/opioid/Code/src/utils/merge_tables.py > ../output/riskslim/county/merge_tables.txt &

### COUNTY SUMMARY FOR REASONING BEHIND THE DIFFERENCE IN LTOUR AND COUNTY TABLE
# nohup python3 /mnt/phd/jihu/opioid/Code/src/utils/county_summary.py 2018 > ../output/riskslim/county/condition_stats_2018.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/src/utils/county_summary.py 2019 > ../output/riskslim/county/condition_stats_2019.txt &

### LTOUR BY COUNTY
#Alameda         Alpine          Amador          Butte           Calaveras       Colusa          Contra Costa    Del Norte       El Dorado       Fresno          Glenn          
# [12] Humboldt        Imperial        Inyo            Kern            Kings           Lake            Lassen          Los Angeles     Madera          Marin           Mariposa       
# [23] Mendocino       Merced          Modoc           Mono            Monterey        Napa            Nevada          Orange          Placer          Plumas          Riverside      
# [34] Sacramento      San Benito      San Bernardino  San Diego       San Francisco   San Joaquin     San Luis Obispo San Mateo       Santa Barbara   Santa Clara     Santa Cruz     
# [45] Shasta          Sierra          Siskiyou        Solano          Sonoma          Stanislaus      Sutter          Tehama          Trinity         Tulare          Tuolumne       
# [56] Ventura         Yolo            Yuba 

# counties=("Alameda" "Alpine" "Amador" "Butte" "Calaveras" "Colusa" "Contra Costa" "Del Norte" "El Dorado" "Fresno" "Glenn")
# counties=("Humboldt" "Imperial" "Inyo" "Kern" "Kings" "Lake" "Lassen" "Los Angeles" "Madera" "Marin" "Mariposa")
# counties=("Mendocino" "Merced" "Modoc" "Mono" "Monterey" "Napa" "Nevada" "Orange" "Placer" "Plumas" "Riverside")
# counties=("Sacramento" "San Benito" "San Bernardino" "San Diego" "San Francisco" "San Joaquin" "San Luis Obispo" "San Mateo" "Santa Barbara" "Santa Clara" "Santa Cruz")
# counties=("Shasta" "Sierra" "Siskiyou" "Solano" "Sonoma" "Stanislaus" "Sutter" "Tehama" "Trinity" "Tulare" "Tuolumne")
# counties=("Ventura" "Yolo" "Yuba")

for county in "${counties[@]}"; do
  nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test tableLTOUR_6 county"$county" > "../output/riskslim/county/Test_LTOUR_$county.txt" &
  # nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py test first tableLTOUR_6 county"$county" > "../output/riskslim/county/Test_naive_LTOUR_$county.txt" &
done
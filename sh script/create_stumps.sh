export OMP_NUM_THREADS=10

# CREATE STUMPS
mkdir -p ../output/stumps

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2018 LTOUR > ../output/stumps/create_stumps_2018.txt & 
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2018 Explore > ../output/stumps/create_stumps_2018_explore.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2019 Explore > ../output/stumps/create_stumps_2019_explore.txt &


# OTHER SUBSET OF PRESCRIPTIONS (FIRST PRESCRIPTION ONLY, UP TO FIRST 180 ONLY)

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2018 Explore first > ../output/stumps/create_stumps_2018_first_explore.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2019 Explore first > ../output/stumps/create_stumps_2019_first_explore.txt &

# nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py 2018 Explore upto180 > ../output/stumps/create_stumps_2018_upto180_explore.txt &

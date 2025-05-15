export OMP_NUM_THREADS=10

mkdir -p ../output/stumps

for year in 2018 2019; do
    nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py $year Explore > ../output/stumps/create_stumps_${year}_explore.txt &
    nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py $year Explore first > ../output/stumps/create_stumps_${year}_first_explore.txt &
done
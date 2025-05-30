export OMP_NUM_THREADS=10

mkdir -p ../output/stumps

for year in 2018 2019; do
    nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py $year > ../output/stumps/create_stumps_${year}.txt &
    nohup python3 /mnt/phd/jihu/opioid/Code/risk_stumps.py $year first > ../output/stumps/create_stumps_${year}_first.txt &
done
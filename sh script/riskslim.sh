export OMP_NUM_THREADS=10
mkdir -p ../output/riskslim

nohup python3 /mnt/phd/jihu/opioid/Code/risk_main.py train single 5 6 1e-15 median feature_random essential1 nodrug > ../output/riskslim/LTOUR_longacting_Sep26_test.txt & 

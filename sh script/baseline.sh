export OMP_NUM_THREADS=10

mkdir -p ../output/baseline/median

# running all these now...
nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py DecisionTree median > ../output/baseline/median/DT_median.txt & 
nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py RandomForest median > ../output/baseline/median/RF_median.txt & 
nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py L1 median > ../output/baseline/median/L1_median.txt & 
nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py L2 median > ../output/baseline/median/L2_median.txt & 
nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py LinearSVM median > ../output/baseline/median/SVM_median.txt & 
nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py XGB median > ../output/baseline/median/XGB_median.txt & 
nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py NN median > ../output/baseline/median/NN_median.txt & 
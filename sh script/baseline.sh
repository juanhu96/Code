export OMP_NUM_THREADS=10
mkdir -p ../output/baseline

# models=("DecisionTree" "RandomForest" "LinearSVM" "XGB" "NN" "L1" "L2")
# declare -A model_short_names=( ["DecisionTree"]="DT" ["RandomForest"]="RF" ["LinearSVM"]="SVM" ["XGB"]="XGB" ["NN"]="NN" ["L1"]="L1" ["L2"]="L2" )
# for model in "${models[@]}"; do
#     short_name="${model_short_names[$model]}"
#     nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py "$model" naive > "../output/baseline/${short_name}.txt" &
# done

# nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py DecisionTree naive > ../output/baseline/DT_test.txt &
nohup python3 /mnt/phd/jihu/opioid/Code/baseline_main.py Logistic bracket naive > ../output/baseline/Logistic_bracket_naive.txt &
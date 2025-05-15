export OMP_NUM_THREADS=10


# STEP 1: Filter and split data

# nohup python3 ../src/process/step1_filter_split_data.py 2018 > ../output/data_process/step1_filter_split_data_2018.txt & # done
# nohup python3 ../src/process/step1_filter_split_data.py 2019 > ../output/data_process/step1_filter_split_data_2019.txt & # done


# STEP 2 & 3: Identify CURES alert & long-term users

# nohup python3 ../src/process/step23_identify_alert_lt_single.py 2018 > ../output/data_process/step23_identify_alert_lt_single_2018.txt & # done
# nohup python3 ../src/process/step23_identify_alert_lt_multiple.py 2018 1 > ../output/data_process/step23_identify_alert_lt_multiple_2018_1.txt & # done
# nohup python3 ../src/process/step23_identify_alert_lt_multiple.py 2018 2 > ../output/data_process/step23_identify_alert_lt_multiple_2018_2.txt & # done
# nohup python3 ../src/process/step23_identify_alert_lt_multiple.py 2018 3 > ../output/data_process/step23_identify_alert_lt_multiple_2018_3.txt & # done
# nohup python3 ../src/process/step23_identify_alert_lt_multiple.py 2018 4 > ../output/data_process/step23_identify_alert_lt_multiple_2018_4.txt & # done

# nohup python3 ../src/process/step23_identify_alert_lt_single.py 2019 > ../output/data_process/step23_identify_alert_lt_single_2019.txt & # done
# nohup python3 ../src/process/step23_identify_alert_lt_multiple.py 2019 1 > ../output/data_process/step23_identify_alert_lt_multiple_2019_1.txt & # done
# nohup python3 ../src/process/step23_identify_alert_lt_multiple.py 2019 2 > ../output/data_process/step23_identify_alert_lt_multiple_2019_2.txt & # done
# nohup python3 ../src/process/step23_identify_alert_lt_multiple.py 2019 3 > ../output/data_process/step23_identify_alert_lt_multiple_2019_3.txt & # done
# nohup python3 ../src/process/step23_identify_alert_lt_multiple.py 2019 4 > ../output/data_process/step23_identify_alert_lt_multiple_2019_4.txt & # done


# STEP 4: Prescription-based feature engineering

# nohup python3 ../src/process/step4_compute_features.py 2018 single > ../output/data_process/step4_compute_features_2018_single.txt & # done
# nohup python3 ../src/process/step4_compute_features.py 2018 1 > ../output/data_process/step4_compute_features_2018_1.txt & # done
# nohup python3 ../src/process/step4_compute_features.py 2018 2 > ../output/data_process/step4_compute_features_2018_2.txt & # done
# nohup python3 ../src/process/step4_compute_features.py 2018 3 > ../output/data_process/step4_compute_features_2018_3.txt & # done
# nohup python3 ../src/process/step4_compute_features.py 2018 4 > ../output/data_process/step4_compute_features_2018_4.txt & # done

# nohup python3 ../src/process/step4_compute_features.py 2019 single > ../output/data_process/step4_compute_features_2019_single.txt & # done
# nohup python3 ../src/process/step4_compute_features.py 2019 1 > ../output/data_process/step4_compute_features_2019_1.txt & # done
# nohup python3 ../src/process/step4_compute_features.py 2019 2 > ../output/data_process/step4_compute_features_2019_2.txt & # done
# nohup python3 ../src/process/step4_compute_features.py 2019 3 > ../output/data_process/step4_compute_features_2019_3.txt & # done
# nohup python3 ../src/process/step4_compute_features.py 2019 4 > ../output/data_process/step4_compute_features_2019_4.txt & # done


# STEP 5: Patient/prescriber-based feature engineering

# nohup python3 ../src/process/step5_compute_features_patient_prescriber.py 2018 > ../output/data_process/step5_compute_features_patient_prescriber_2018.txt & # done
# nohup python3 ../src/process/step5_compute_features_patient_prescriber.py 2019 > ../output/data_process/step5_compute_features_patient_prescriber_2019.txt & # done


# STEP 6: Convert to input format

# nohup python3 ../src/process/step6_to_input.py 2018 > ../output/data_process/step6_to_input_2018.txt & # done
# nohup python3 ../src/process/step6_to_input.py 2019 > ../output/data_process/step6_to_input_2019.txt & # done

nohup python3 ../src/process/step6_to_input.py 2018 > ../output/data_process/step6_to_input_first_2018.txt & # running
nohup python3 ../src/process/step6_to_input.py 2019 > ../output/data_process/step6_to_input_first_2019.txt & # running


# STEP 7: Final check

# nohup python3 ../src/process/step7_final_check.py 2018 > ../output/data_process/step7_final_check_2018.txt & # done
# nohup python3 ../src/process/step7_final_check.py 2019 > ../output/data_process/step7_final_check_2019.txt & # done


# SUMMARY
# nohup python3 /mnt/phd/jihu/opioid/Code/src/summary_stats.py 2018 > ../output/data_process/summary_stats_2018.txt &
# nohup python3 /mnt/phd/jihu/opioid/Code/src/summary_stats.py 2019 > ../output/data_process/summary_stats_2019.txt &
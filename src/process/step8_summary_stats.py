'''
STEP 8

INPUT: FULL_OPIOID_2018_INPUT.csv, FULL_OPIOID_2019_INPUT.csv
OUTPUT: SUMMARY TABLE IN LATEX FORM
'''

import pandas as pd
import numpy as np
datadir = "/export/storage_cures/CURES/Processed/"
resultdir = "/export/storage_cures/CURES/Results/"

FULL_INPUT_2018 = pd.read_csv(f"{datadir}FULL_OPIOID_2018_INPUT.csv")
FULL_INPUT_2019 = pd.read_csv(f"{datadir}FULL_OPIOID_2019_INPUT.csv")
FULL_INPUT = pd.concat([FULL_INPUT_2018, FULL_INPUT_2019], ignore_index=True)

# -------------------------------
# PATIENT-LEVEL STATS
# -------------------------------
PATIENTS = (
    FULL_INPUT.groupby('patient_id')
    .agg(
        gender=('patient_gender', 'first'),
        age=('age', 'first'),
        num_prescriptions=('patient_id', 'count'),
        concurrent_benzo=('concurrent_benzo', lambda x: int(x.sum() > 0)),
        long_term_user=('long_term_180', lambda x: int(x.sum() > 0)),
        HPI_quartile=('patient_HPIQuartile', 'first')
    )
    .reset_index()
)

LT = FULL_INPUT[FULL_INPUT['patient_id'].isin(PATIENTS[PATIENTS['long_term_user'] == 1]['patient_id'])]
NLT = FULL_INPUT[FULL_INPUT['patient_id'].isin(PATIENTS[PATIENTS['long_term_user'] == 0]['patient_id'])]

# -------------------------------
# GENERATE LATEX TABLE CONTENT
# -------------------------------
def fmt_num(n):
    return f"{n:,}"

def fmt_pct(count, total):
    return f"{fmt_num(count)} ({count / total:.1%})"

def fmt_mean_sd(series):
    return f"{round(series.mean(skipna=True),1)} ({round(series.std(skipna=True),1)})"

def fmt_pct_safe(count, total):
    return f"{fmt_num(count)} ($<0.1\%$)" if count / total < 0.001 else fmt_pct(count, total)

# Total patients
total_patients = len(PATIENTS)
lt_patients = PATIENTS['long_term_user'].sum()
nlt_patients = total_patients - lt_patients

# Total prescriptions
total_rx = len(FULL_INPUT)
lt_rx = len(LT)
nlt_rx = len(NLT)

# Drug and payment
drugs = ['Hydrocodone', 'Oxycodone', 'Codeine', 'Morphine', 'Hydromorphone', 'Methadone', 'Fentanyl', 'Oxymorphone']
payments = ['CommercialIns', 'CashCredit', 'Medicare', 'Medicaid', 'MilitaryIns', 'WorkersComp']

# latex_lines = [
#     r"\begin{table}[p] \centering \sffamily",
#     r"\caption{Summary statistics, 2018-2019} \label{tab:summary_stats_full}",
#     r"\vspace*{0.1cm}",
#     r"\begin{tabular}{lccc}",
#     r"\toprule",
#     r"Variable & All & Non-long-term & Long-term\\",
#     r"\midrule",
#     r"\textbf{Patient characteristics} \\\\",
#     f"Total patients, no. & {fmt_num(total_patients)} & {fmt_pct(nlt_patients, total_patients)} & {fmt_pct(lt_patients, total_patients)} \\\\",
#     f"\\hspace{{0.25cm}} Male & {fmt_pct((PATIENTS['gender']==0).sum(), total_patients)} & {fmt_pct(((PATIENTS['gender']==0)&(PATIENTS['long_term_user']==0)).sum(), nlt_patients)} & {fmt_pct(((PATIENTS['gender']==0)&(PATIENTS['long_term_user']==1)).sum(), lt_patients)} \\\\",
#     f"\\hspace{{0.25cm}} Female & {fmt_pct((PATIENTS['gender']==1).sum(), total_patients)} & {fmt_pct(((PATIENTS['gender']==1)&(PATIENTS['long_term_user']==0)).sum(), nlt_patients)} & {fmt_pct(((PATIENTS['gender']==1)&(PATIENTS['long_term_user']==1)).sum(), lt_patients)} \\\\",
#     f"Age, mean (SD) & {fmt_mean_sd(PATIENTS['age'])} & {fmt_mean_sd(PATIENTS[PATIENTS['long_term_user']==0]['age'])} & {fmt_mean_sd(PATIENTS[PATIENTS['long_term_user']==1]['age'])} \\\\",
#     f"Opioid Rx per patient, mean (SD) & {fmt_mean_sd(PATIENTS['num_prescriptions'])} & {fmt_mean_sd(PATIENTS[PATIENTS['long_term_user']==0]['num_prescriptions'])} & {fmt_mean_sd(PATIENTS[PATIENTS['long_term_user']==1]['num_prescriptions'])} \\\\",
#     f"Concurrent Benzodiazepine, \\% & {fmt_pct(PATIENTS['concurrent_benzo'].sum(), total_patients)} & {fmt_pct(PATIENTS[PATIENTS['long_term_user']==0]['concurrent_benzo'].sum(), nlt_patients)} & {fmt_pct(PATIENTS[PATIENTS['long_term_user']==1]['concurrent_benzo'].sum(), lt_patients)} \\\\",
#     r"HPI Quartile \\\\",
#     *[
#         f"\\hspace{{0.25cm}} {label} & {fmt_pct((PATIENTS['HPI_quartile']==q).sum(), total_patients)} & {fmt_pct(((PATIENTS['HPI_quartile']==q)&(PATIENTS['long_term_user']==0)).sum(), nlt_patients)} & {fmt_pct(((PATIENTS['HPI_quartile']==q)&(PATIENTS['long_term_user']==1)).sum(), lt_patients)} \\\\"
#         for q, label in zip([1,2,3,4], ["Bottom 25\\%", "25\\% - 50\\%", "50\\% - 75\\%", "Top 25\\%"])
#     ],
#     r"\\",
#     r"\textbf{Prescription characteristics} \\\\",
#     f"Total prescriptions, no. & {fmt_num(total_rx)} & {fmt_pct(nlt_rx, total_rx)} & {fmt_pct(lt_rx, total_rx)} \\\\",
#     f"Daily MME per Rx, mean (SD) & {fmt_mean_sd(FULL_INPUT['daily_dose'])} & {fmt_mean_sd(NLT['daily_dose'])} & {fmt_mean_sd(LT['daily_dose'])} \\\\",
#     f"Pills dispensed per Rx, mean (SD) & {fmt_mean_sd(FULL_INPUT['quantity'])} & {fmt_mean_sd(NLT['quantity'])} & {fmt_mean_sd(LT['quantity'])} \\\\",
#     f"Days supply per Rx, mean (SD) & {fmt_mean_sd(FULL_INPUT['days_supply'])} & {fmt_mean_sd(NLT['days_supply'])} & {fmt_mean_sd(LT['days_supply'])} \\\\",
#     f"Long acting opioid, \\% & {fmt_pct((FULL_INPUT['long_acting']==1).sum(), total_rx)} & {fmt_pct((NLT['long_acting']==1).sum(), nlt_rx)} & {fmt_pct((LT['long_acting']==1).sum(), lt_rx)} \\\\",
# ]


latex_lines = [
    r"\begin{table}[p] \centering \sffamily",
    r"\caption{Summary statistics, 2018-2019} \label{tab:summary_stats_full}",
    r"\vspace*{0.1cm}",
    r"\begin{tabular}{lccc}",
    r"\toprule",
    r"Variable & All & Non-long-term & Long-term\\",
    r"\midrule",
    r"\textbf{Patient characteristics} \\",
    f"Total patients, no. & {fmt_num(total_patients)} & {fmt_pct(nlt_patients, total_patients)} & {fmt_pct(lt_patients, total_patients)} \\",
    f"\hspace{{0.25cm}} Male & {fmt_pct((PATIENTS['gender']==0).sum(), total_patients)} & {fmt_pct(((PATIENTS['gender']==0)&(PATIENTS['long_term_user']==0)).sum(), nlt_patients)} & {fmt_pct(((PATIENTS['gender']==0)&(PATIENTS['long_term_user']==1)).sum(), lt_patients)} \\",
    f"\hspace{{0.25cm}} Female & {fmt_pct((PATIENTS['gender']==1).sum(), total_patients)} & {fmt_pct(((PATIENTS['gender']==1)&(PATIENTS['long_term_user']==0)).sum(), nlt_patients)} & {fmt_pct(((PATIENTS['gender']==1)&(PATIENTS['long_term_user']==1)).sum(), lt_patients)} \\",
    f"Age, mean (SD) & {fmt_mean_sd(PATIENTS['age'])} & {fmt_mean_sd(PATIENTS[PATIENTS['long_term_user']==0]['age'])} & {fmt_mean_sd(PATIENTS[PATIENTS['long_term_user']==1]['age'])} \\",
    f"Opioid Rx per patient, mean (SD) & {fmt_mean_sd(PATIENTS['num_prescriptions'])} & {fmt_mean_sd(PATIENTS[PATIENTS['long_term_user']==0]['num_prescriptions'])} & {fmt_mean_sd(PATIENTS[PATIENTS['long_term_user']==1]['num_prescriptions'])} \\",
    f"Concurrent Benzodiazepine, \% & {fmt_pct(PATIENTS['concurrent_benzo'].sum(), total_patients)} & {fmt_pct(PATIENTS[PATIENTS['long_term_user']==0]['concurrent_benzo'].sum(), nlt_patients)} & {fmt_pct(PATIENTS[PATIENTS['long_term_user']==1]['concurrent_benzo'].sum(), lt_patients)} \\",
    r"HPI Quartile \\" ,
    *[
        f"\hspace{{0.25cm}} {label} & {fmt_pct((PATIENTS['HPI_quartile']==q).sum(), total_patients)} & {fmt_pct(((PATIENTS['HPI_quartile']==q)&(PATIENTS['long_term_user']==0)).sum(), nlt_patients)} & {fmt_pct(((PATIENTS['HPI_quartile']==q)&(PATIENTS['long_term_user']==1)).sum(), lt_patients)} \\" 
        for q, label in zip([1,2,3,4], ["Bottom 25\%", "25\% - 50\%", "50\% - 75\%", "Top 25\%"])
    ],
    r"\\",
    r"\textbf{Prescription characteristics} \\" ,
    f"Total prescriptions, no. & {fmt_num(total_rx)} & {fmt_pct(nlt_rx, total_rx)} & {fmt_pct(lt_rx, total_rx)} \\",
    f"Daily MME per Rx, mean (SD) & {fmt_mean_sd(FULL_INPUT['daily_dose'])} & {fmt_mean_sd(NLT['daily_dose'])} & {fmt_mean_sd(LT['daily_dose'])} \\",
    f"Pills dispensed per Rx, mean (SD) & {fmt_mean_sd(FULL_INPUT['quantity'])} & {fmt_mean_sd(NLT['quantity'])} & {fmt_mean_sd(LT['quantity'])} \\",
    f"Days supply per Rx, mean (SD) & {fmt_mean_sd(FULL_INPUT['days_supply'])} & {fmt_mean_sd(NLT['days_supply'])} & {fmt_mean_sd(LT['days_supply'])} \\",
    f"Long acting opioid, \% & {fmt_pct((FULL_INPUT['long_acting']==1).sum(), total_rx)} & {fmt_pct((NLT['long_acting']==1).sum(), nlt_rx)} & {fmt_pct((LT['long_acting']==1).sum(), lt_rx)} \\",
    r"Opiate prescribed, \%  \\" ,
    *[
        f"\hspace{{0.25cm}} {drug} & {fmt_pct_safe((FULL_INPUT[drug]==1).sum(), total_rx)} & {fmt_pct_safe((NLT[drug]==1).sum(), nlt_rx)} & {fmt_pct_safe((LT[drug]==1).sum(), lt_rx)} \\" 
        for drug in drugs
    ],
    r"Payment type, \% \\" ,
    *[
        f"\hspace{{0.25cm}} {ptype.replace('CashCredit', 'Cash or Credit').replace('CommercialIns', 'Commercial Insurance').replace('MilitaryIns', 'Military Insurance').replace('WorkersComp', 'Workers Compensation')} & {fmt_pct((FULL_INPUT[ptype]==1).sum(), total_rx)} & {fmt_pct((NLT[ptype]==1).sum(), nlt_rx)} & {fmt_pct((LT[ptype]==1).sum(), lt_rx)} \\" 
        for ptype in payments
    ] + [
        f"\hspace{{0.25cm}} Other & {fmt_pct(((FULL_INPUT['Other']==1)|(FULL_INPUT['IndianNation']==1)).sum(), total_rx)} & {fmt_pct(((NLT['Other']==1)|(NLT['IndianNation']==1)).sum(), nlt_rx)} & {fmt_pct(((LT['Other']==1)|(LT['IndianNation']==1)).sum(), lt_rx)} \\"
    ],
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}"
]


# Export LaTeX table
with open(f"{resultdir}summary_stats.tex", "w") as f:
    f.write("\n".join(latex_lines + [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ]))

print("LaTeX summary table saved to summary_stats_table.tex")

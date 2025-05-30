import os
import sys
import re
from collections import defaultdict
datadir = "/export/storage_cures/CURES/Processed/"
resultdir = "/export/storage_cures/CURES/Results/"

All = False
County = True

if All:
    # ==== Settings ====
    table_names = ['CA_2018', 'CA_2019'] # 2018 vs. 2019
    column_indices_to_merge = [2, 3]  # which columns to extract from each LaTeX table (0-based)

    # ==== Prepare ====
    table_files = [f'summary_stats_{name}.tex' for name in table_names]
    table_paths = [os.path.join(resultdir, fname) for fname in table_files]
    table_labels = [re.search(r'summary_stats_(.*).tex', f).group(1).replace('_', ' ') for f in table_files]

    # ==== Extract Data ====
    all_data = defaultdict(dict)
    row_order = []

    for filepath, label in zip(table_paths, table_labels):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if '&' not in line or '\\' not in line or line.startswith('%'):
                continue

            parts = [p.strip() for p in line.split('&')]
            if len(parts) < max(column_indices_to_merge) + 1:
                continue

            varname = parts[0]
            if varname in ['Variable', '']:
                continue

            if varname not in row_order:
                row_order.append(varname)

            merged = " ".join(parts[i] for i in column_indices_to_merge)
            # merged = merged.replace('%', r'\%')
            all_data[varname][label] = merged

    # ==== Build LaTeX Table ====
    latex_lines = [
        r"\begin{table}[p] \centering \sffamily",
        r"\caption{Summary statistics across time periods} \label{tab:summary_stats_multi}",
        r"\vspace*{0.1cm}",
        r"\begin{tabular}{l" + "c" * len(table_labels) + "}",
        r"\toprule",
        "Variable & " + " & ".join(table_labels) + r" \\",
        r"\midrule"
    ]

    for var in row_order:
        values = [all_data[var].get(label, '') for label in table_labels]
        latex_lines.append(var + " & " + " & ".join(values) + r" \\")

    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ]

    # ==== Save Output ====
    output_path = os.path.join(resultdir, "summary_stats_merged.tex")
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))

    print("\n".join(latex_lines))
    print(f"Merged LaTeX table saved to: {output_path}")



if County:

    # ==== Settings ====
    table_names = ['Fresno', 'San Bernardino', 'Los Angeles', 'Humboldt', 'San Benito', 'Riverside']  # counties
    column_indices_to_merge = [1]  # which columns to extract from each LaTeX table (0-based)

    # ==== Prepare ====
    table_files = [f'summary_stats_{name}_total.tex' for name in table_names]
    table_paths = [os.path.join(resultdir, fname) for fname in table_files]
    table_labels = [name for name in table_names]

    # ==== Extract Data ====
    all_data = defaultdict(dict)
    row_order = []

    for filepath, label in zip(table_paths, table_labels):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if '&' not in line or '\\' not in line or line.startswith('%'):
                continue

            parts = [p.strip() for p in line.split('&')]
            if len(parts) < max(column_indices_to_merge) + 1:
                continue

            varname = parts[0]
            if varname in ['Variable', '']:
                continue

            if varname not in row_order:
                row_order.append(varname)

            merged = " ".join(parts[i] for i in column_indices_to_merge)
            all_data[varname][label] = merged

    # ==== Build LaTeX Table ====
    latex_lines = [
        r"\begin{table}[p] \centering \sffamily",
        r"\caption{Summary statistics across counties} \label{tab:summary_stats_counties}",
        r"\vspace*{0.1cm}",
        r"\begin{tabular}{l" + "c" * len(table_labels) + "}",
        r"\toprule",
        "Variable & " + " & ".join(table_labels) + r" \\",
        r"\midrule"
    ]

    for var in row_order:
        values = [all_data[var].get(label, '') for label in table_labels]
        latex_lines.append(var + " & " + " & ".join(values) + r" \\")

    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ]

    # ==== Save Output ====
    output_path = os.path.join(resultdir, "summary_stats_merged_counties.tex")
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))

    print("\n".join(latex_lines))
    print(f"Merged LaTeX table saved to: {output_path}")
import os
import sys
import re
from collections import defaultdict
datadir = "/export/storage_cures/CURES/Processed/"
resultdir = "/export/storage_cures/CURES/Results/"

county_names = ['Kern', 'Los Angeles', 'Riverside', 'San Bernardino', 'San Francisco', 'San Mateo']

county_files = [f'summary_stats_{name}.tex' for name in county_names]
county_paths = [os.path.join(resultdir, fname) for fname in county_files]
county_labels = [re.search(r'summary_stats_(.*).tex', f).group(1).replace('_', ' ') for f in county_files]
all_data = defaultdict(dict)

# Extract rows
for fname, county in zip(county_paths, county_labels):
    with open(fname, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if '&' in line and not line.startswith('%') and '\\' in line:
            parts = [p.strip() for p in line.split('&')]
            varname = parts[0]
            val = parts[1] if len(parts) > 1 else ''
            all_data[varname][county] = val

# Build merged LaTeX table
header = r"\begin{table}[p] \centering \sffamily" + "\n"
header += r"\caption{Summary statistics across counties} \label{tab:summary_stats_multi}" + "\n"
header += r"\vspace*{0.1cm}" + "\n"
header += r"\begin{tabular}{l" + "c" * len(county_labels) + "}" + "\n"
header += r"\toprule" + "\n"
header += "Variable & " + " & ".join(county_labels) + r"\\" + "\n"
header += r"\midrule" + "\n"

rows = []
for var in all_data:
    row = var
    for county in county_labels:
        row += " & " + all_data[var].get(county, '')
    row += r" \\"
    rows.append(row)

footer = r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"


with open(f"{resultdir}summary_stats_county_merged.tex", "w") as f:
    f.write(header + "\n".join(rows) + "\n" + footer)
print("Merged LaTeX summary saved to summary_stats_county_merged.tex")

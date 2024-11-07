import pandas as pd
import matplotlib.pyplot as plt
import argparse 

parser = argparse.ArgumentParser(description='Extract wall time from .out files and combine into a CSV.')
parser.add_argument('csv', type=str, help='CSV output')
parser.add_argument('output_png', type=str, help='png')
args = parser.parse_args()
csv = args.csv
output_png = args.output_png


df = pd.read_csv(csv)

# Plot the data
plt.figure(figsize=(10,6))


grouped = df.groupby('Number of Elements')
for elems, group in grouped:
    plt.plot(group['Number of DPUs'], group['Walltime [sec]'], marker='o', label=f'Elements = {elems}')

# Adding labels and title
plt.title('Wall Time vs Number of DPUs')
plt.xlabel('Number of DPUs')
plt.ylabel('Wall Time (sec)')
plt.legend(title='Number of Elements', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(output_png)

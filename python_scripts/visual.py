import pandas as pd
import matplotlib.pyplot as plt
import argparse 

parser = argparse.ArgumentParser(description='Extract wall time from .out files and combine into a CSV.')
parser.add_argument('model', type=str, help='programming model')
parser.add_argument('csv', type=str, help='CSV output')
parser.add_argument('output_png', type=str, help='png')
args = parser.parse_args()
csv = args.csv
output_png = args.output_png


df = pd.read_csv(csv)
plt.figure(figsize=(10,6))

# drop columns
df = df.drop(columns=['Wall Time [hour:min:sec]', 'Filename', 'Benchmark'])

# group by trial
grouped = df.groupby(['Number of Elements', 'Number of DPUs']).agg({'Walltime [sec]': 'mean'}).reset_index()

for elem, group in grouped.groupby('Number of Elements'):
    plt.plot(group['Number of DPUs'], group['Walltime [sec]'], marker='o', label=f'{elem}')

title = f"{args.model} : Wall Time vs DPUs"
plt.title(title)
plt.xlabel('Number of DPUs')
plt.ylabel('Wall Time (sec)')
plt.legend(title='Number of Elements', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(output_png)

print(f"plot {output_png} created.")

import os
import re
import csv
import argparse


parser = argparse.ArgumentParser(description='Extract wall time from .out files and combine into a CSV.')
parser.add_argument('input_dir', type=str, help='Directory containing the .out files')
parser.add_argument('output_csv', type=str, help='CSV output')

args = parser.parse_args()


input_dir = args.input_dir
output_file = args.output_csv
output_csv = f"{input_dir}/{output_file}"

pattern = r'(?P<elapsed>\d+:\d+\.\d+)elapsed'
pattern_file = r'(?P<benchmark>[a-zA-Z0-9_]+)_dpu(?P<num_dpus>\d+)elem(?P<num_elems>\d+)\.out'
results = []

# Iterate through each .out file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".out"):
        matchfile = re.search(pattern_file, filename)
        if matchfile:
            datafile = matchfile.groupdict()
            print(datafile)                            
            with open(os.path.join(input_dir, filename), 'r') as file:
                content = file.read()
                match = re.search(pattern, content)
                if match:
                    wall_time = match.group(1)
                    minutes, seconds = map(float, wall_time.split(':'))
                    total_seconds = (minutes * 60) + seconds
                    results.append([filename, datafile["benchmark"], datafile["num_dpus"], datafile["num_elems"], wall_time, total_seconds])

            # Write the data to a CSV file
            with open(output_csv, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["Filename", "Benchmark","Number of DPUs", "Number of Elements", "Wall Time [hour:min:sec]", "Walltime [sec]"])
                csv_writer.writerows(results)
print(f"CSV file '{output_csv}' created with wall time data.")


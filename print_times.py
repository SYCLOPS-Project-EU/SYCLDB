import re
import numpy as np
import matplotlib.pyplot as plt

# Initialize empty lists to store the extracted times and deviations
times = []
deviations = []

# Open the file and read line by line
with open('sycldb_results2.txt', 'r') as file:
    for line in file:
        # Use regular expression to find lines starting with the required pattern
        match = re.match(r"Hot runs: average probe kernels time \(9 runs\): ([\d\.]+) ms \(\+\-([\d\.]+)%\)", line)
        if match:
            # Extract time in ms and deviation
            time_value = float(match.group(1))  # the time in ms
            deviation_value = float(match.group(2))  # the deviation in percentage

            # Append to the respective lists
            times.append(time_value)
            deviations.append(deviation_value)

# Convert lists to NumPy arrays
times_array = np.array(times)
# multiply each deviation by the time
deviations_array = np.array(deviations) / 100 * times_array

# Print or further process the arrays
print("Times in ms:", times_array)
print("Deviations in %:", deviations_array)

# plot the first 13 values, then the next 13 values, and so on
# use instead bar chart and times next to each other
for i in range(0, len(times_array), 13):
    plt.bar(range(i, i + 13), times_array[i:i + 13], yerr=deviations_array[i:i + 13], capsize=5)
    plt.xticks(range(i, i + 13), range(i, i + 13))
    plt.xlabel('Run number')
    plt.ylabel('Time (ms)')
    plt.title('Times and deviations for the first 13 runs')
plt.savefig('times_plot.png')

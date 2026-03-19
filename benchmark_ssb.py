import os
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import statistics

# Configuration
queries = ["q11", "q12", "q13", "q21", "q22", "q23", "q31", "q32", "q33", "q34", "q41", "q42", "q43"]
targets = [1, 2, 3, 4]
target_names = {1: "CPU (Intel)", 2: "NVIDIA L40S", 3: "AMD MI210", 4: "Intel GPU Flex"}
compilers = ["icpx", "acpp"]
modes = [0, 1]

# 1. Compile
print("Compiling all SSB queries for both compilers in parallel...")
all_both = [f"ssb/{q}_both" for q in queries]
cmd = ["make", "-j13"] + all_both
subprocess.run(cmd)

# 2. Run and benchmark
acpp_lib = "/media/ACPP/AdaptiveCpp-25.10.0/install/lib"
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = acpp_lib + ":" + env.get("LD_LIBRARY_PATH", "")

# Struct to hold data: results[device][query][mode][compiler] = time
results_data = {target_names[t]: {q: {m: {comp: 0.0 for comp in compilers} for m in modes} for q in queries} for t in targets}

for q in queries:
    for m in modes:
        for comp in compilers:
            binary = f"ssb/{q}_{comp}"
            if not os.path.exists(binary):
                continue
            for t in targets:
                print(f"Running {binary} -t {t} -m {m}...")
                cmd = [f"./{binary}", "-t", str(t), "-r", "3", "-m", str(m)]
                try:
                    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=180)
                    if res.returncode == 0:
                        if m == 1:
                            matches = re.findall(r'External total timer reported\s+([\d.]+)\s*ms', res.stdout)
                        else:
                            matches = re.findall(r'Kernel Time:\s+([\d.]+)\s*ms', res.stdout)
                        
                        if matches:
                            results_data[target_names[t]][q][m][comp] = statistics.median([float(x) for x in matches])
                except subprocess.TimeoutExpired:
                    print(f"TIMEOUT: {binary} on {target_names[t]} (mode {m})")

# 3. Plotting
print("Generating plot...")
fig, axes = plt.subplots(4, 1, figsize=(16, 24))
width = 0.2
x = np.arange(len(queries))
device_list = [target_names[t] for t in targets]

for i, dev in enumerate(device_list):
    ax = axes[i]
    icpx_h = [results_data[dev][q][0]["icpx"] for q in queries]
    acpp_h = [results_data[dev][q][0]["acpp"] for q in queries]
    icpx_c = [results_data[dev][q][1]["icpx"] for q in queries]
    acpp_c = [results_data[dev][q][1]["acpp"] for q in queries]
    
    ax.bar(x - 1.5*width, icpx_h, width, label='ICPX Hardcoded', color='skyblue')
    ax.bar(x - 0.5*width, acpp_h, width, label='ACPP Hardcoded', color='dodgerblue')
    ax.bar(x + 0.5*width, icpx_c, width, label='ICPX Pipelined', color='salmon')
    ax.bar(x + 1.5*width, acpp_c, width, label='ACPP Pipelined', color='red')
    
    ax.set_title(f'SSB Benchmark Performance on {dev}', fontsize=16)
    ax.set_ylabel('Execution Time (ms)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(queries, fontsize=12)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)

plt.tight_layout()
plt.savefig('ssb_compiler_comparison.png', dpi=300)
print("Plot saved as ssb_compiler_comparison.png")

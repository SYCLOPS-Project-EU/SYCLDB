import os
import subprocess
import re
import statistics
import sys

# Set ACPP path
acpp_lib = "/media/ACPP/AdaptiveCpp-25.10.0/install/lib"
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = acpp_lib + ":" + env.get("LD_LIBRARY_PATH", "")

def run_cmd(cmd):
    try:
        # Some binaries are in ssb/
        binary = cmd[0]
        if not os.path.exists(binary):
            return f"Error: Binary {binary} not found"
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=120)
        return res.stdout
    except Exception as e:
        return f"Exception: {str(e)}"

def get_time(out):
    times = []
    # Pattern: Repetition X Kernel Time: Y ms
    # Pattern: Repetition X | Project time: Y ms
    # Pattern: Repetition X | Build time: Y ms | Probe time: Z ms
    
    m_kt = re.findall(r'Kernel Time:\s+([\d.]+)\s*ms', out)
    if m_kt: times.extend([float(x) for x in m_kt])
    
    m_pt = re.findall(r'Project time:\s+([\d.]+)\s*ms', out)
    if m_pt: times.extend([float(x) for x in m_pt])
    
    m_probe = re.findall(r'Probe time:\s+([\d.]+)\s*ms', out)
    if m_probe: times.extend([float(x) for x in m_probe])
    
    if times: return statistics.median(times)
    return 0.0

# 1. Benchmark Project/Join
print("Benchmarking Project/Join...")
results = []
for comp in ["icpx", "acpp"]:
    # Project
    p_old = get_time(run_cmd([f"./project_{comp}", "-t", "1", "-m", "0", "-r", "5"]))
    p_new = get_time(run_cmd([f"./project_{comp}", "-t", "1", "-m", "1", "-r", "5"]))
    results.append({"Bench": "Project", "Comp": comp.upper(), "Old": p_old, "New": p_new})
    
    # Join
    j_old = get_time(run_cmd([f"./join_{comp}", "-t", "1", "-m", "0", "-r", "3"]))
    j_new = get_time(run_cmd([f"./join_{comp}", "-t", "1", "-m", "1", "-r", "3"]))
    results.append({"Bench": "Join Probe", "Comp": comp.upper(), "Old": j_old, "New": j_new})

# 2. SSB Queries
print("Benchmarking SSB...")
ssb_queries = ["q11"] # Just do q11 for representative sample
if "--all" in sys.argv:
    ssb_queries = ["q11", "q12", "q13", "q21", "q22", "q23", "q31", "q32", "q33", "q34", "q41", "q42", "q43"]

for q in ssb_queries:
    for comp in ["icpx", "acpp"]:
        binary = f"./ssb/{q}_{comp}"
        t_old = get_time(run_cmd([binary, "-t", "1", "-m", "0", "-r", "3"]))
        t_new = get_time(run_cmd([binary, "-t", "1", "-m", "0", "-O", "1", "-r", "3"]))
        results.append({"Bench": q, "Comp": comp.upper(), "Old": t_old, "New": t_new})

# Report
out_str = "## CPU Strategy Comparison Results\n\n"
out_str += "| Benchmark | Compiler | Old Strategy (ms) | New Strategy (ms) | Speedup |\n"
out_str += "|---|---|---|---|---|\n"
for res in results:
    s = res["Old"] / res["New"] if res["New"] > 0 else 0
    out_str += f"| {res['Bench']} | {res['Comp']} | {res['Old']:.2f} | {res['New']:.2f} | {s:.2f}x |\n"

print(out_str)
with open("cpu_strategy_comparison.md", "w") as f:
    f.write(out_str)

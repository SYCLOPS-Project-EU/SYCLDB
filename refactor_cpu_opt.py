import os
import re

def refactor_file(filepath):
    print(f"Refactoring {filepath}...")
    with open(filepath, 'r') as f:
        content = f.read()

    os.system(f"git checkout {filepath} > /dev/null 2>&1")
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Main - add O
    if 'int optimize = 0;' not in content:
        content = content.replace('int modes = 1;', 'int modes = 1;\n  int optimize = 0;')
        content = content.replace('int modes = 0;', 'int modes = 0;\n  int optimize = 0;')
        content = content.replace('getopt(argc, argv, "t:r:m:")', 'getopt(argc, argv, "t:r:m:O:")')
        content = re.sub(r"(case\s+'m':.*?break;)", r"\1\n    case 'O': optimize = atoi(optarg); break;", content, flags=re.DOTALL)
    
    # 2. Function Signatures (Add use_sharding before idx)
    content = re.sub(
        r',\s*(?:(?://|/\*).*?\n\s*)?sycl::id<1>\s+idx\s*\)',
        r', bool use_sharding, sycl::id<1> idx)',
        content
    )

    # 3. probe_function scope
    content = re.sub(
        r'prob\.probe_function\s*=\s*\[&\]\s*\((.*?)\)\s*\{',
        r'prob.probe_function = [&](\1) {\n    bool use_sharding = queue.get_device().is_cpu() && (optimize == 1);',
        content, flags=re.DOTALL
    )

    # 4. Call Sites (Add use_sharding before idx)
    # Search for (..., idx) or (..., idx) inside the probe_function
    # I'll use a broad approach: replace ', idx)' with ', use_sharding, idx)' 
    # BUT only if followed by } ); or similar closure of a lambda.
    
    probe_start_idx = content.find("prob.probe_function =")
    if probe_start_idx != -1:
        probe_block = content[probe_start_idx:]
        # Use regex for call sites to handle newlines
        new_probe_block = re.sub(r',\s*sycl::id<1>\(idx\)\s*\)', r', use_sharding, idx)', probe_block)
        new_probe_block = re.sub(r',\s*idx\s*\)', r', use_sharding, idx)', new_probe_block)
        content = content[:probe_start_idx] + new_probe_block

    # 5. Atomic indexing
    if 'prob.res_size = 1;' in content:
        content = content.replace('&revenue[0]', '&revenue[use_sharding ? (idx[0]/1024)%1024 : 0]')
        content = content.replace('&res[0]', '&res[use_sharding ? (idx[0]/1024)%1024 : 0]')

    # 6. run_benchmark
    content = content.replace('repetitions, cpu_queue);', 'repetitions, cpu_queue, optimize == 1);')

    with open(filepath, 'w') as f:
        f.write(content)

queries = ["q11", "q12", "q13", "q21", "q22", "q23", "q31", "q32", "q33", "q34", "q41", "q42", "q43"]
for q in queries:
    path = f"ssb/{q}.cpp"
    if os.path.exists(path):
        refactor_file(path)

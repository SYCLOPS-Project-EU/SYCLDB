## CPU Strategy Comparison Results

| Benchmark | Compiler | Old Strategy (ms) | New Strategy (ms) | Speedup |
|---|---|---|---|---|
| Project | ICPX | 0.64 | 0.61 | 1.04x |
| Join Probe | ICPX | 108.31 | 950.86 | 0.11x |
| Project | ACPP | 0.78 | 0.42 | 1.84x |
| Join Probe | ACPP | 94.48 | 5258.83 | 0.02x |
| q11 | ICPX | 161.42 | 11.53 | 14.00x |
| q11 | ACPP | 147.36 | 10.16 | 14.50x |

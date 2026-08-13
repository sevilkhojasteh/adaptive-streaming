import numpy as np
import os

os.makedirs("generated", exist_ok=True)

np.random.seed(42)

for i in range(10):
    trace = np.random.normal(loc=3000, scale=1500, size=300)
    trace = np.clip(trace, 300, 8000)

    with open(f"generated/trace_{i}.txt", "w") as f:
        for t in trace:
            f.write(f"0 {t/1000}\n")  # mimic Pensieve format (Mbps)

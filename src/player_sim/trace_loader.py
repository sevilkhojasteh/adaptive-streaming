import os

def load_trace(filepath):
    trace = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                bandwidth_mbps = float(parts[1])
                trace.append(bandwidth_mbps * 1000)  # convert to kbps
    return trace

def load_random_trace(trace_folder):
    files = [f for f in os.listdir(trace_folder) if f.endswith(".txt")]

    if not files:
        raise ValueError("No trace files found.")

    file = os.path.join(trace_folder, files[0])
    return load_trace(file)
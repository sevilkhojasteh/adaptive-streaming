import numpy as np
from abr_simulator import ABRSimulator

# Simulated network trace (kbps)
network_trace = np.random.uniform(500, 6000, 30)

bitrates = [400, 800, 1500, 3000, 6000]

sim = ABRSimulator(bitrates)
history = sim.simulate(network_trace)

print("Total Rebuffer Time:", sim.rebuffer_time)
print("Bitrate Switches:", sim.switches)

for entry in history[:5]:
    print(entry)

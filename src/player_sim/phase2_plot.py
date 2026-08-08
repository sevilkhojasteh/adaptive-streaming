import matplotlib.pyplot as plt
import numpy as np

from abr_simulator import ABRSimulator
from metrics import QoEModel

np.random.seed(42)
network_trace = np.random.uniform(500, 6000, 60)

bitrates = [400, 800, 1500, 3000, 6000]

sim_rate = ABRSimulator(bitrates)
hist_rate = sim_rate.simulate(network_trace, mode="rate")

sim_buffer = ABRSimulator(bitrates)
hist_buffer = sim_buffer.simulate(network_trace, mode="buffer")

qoe_model = QoEModel()

qoe_rate = qoe_model.compute(hist_rate, sim_rate.rebuffer_time)
qoe_buffer = qoe_model.compute(hist_buffer, sim_buffer.rebuffer_time)

print("Rate QoE:", qoe_rate)
print("Buffer QoE:", qoe_buffer)

# Plot
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(network_trace)
plt.title("True Bandwidth (kbps)")

plt.subplot(3,1,2)
plt.plot([h["bitrate"] for h in hist_rate], label="Rate-Based")
plt.plot([h["bitrate"] for h in hist_buffer], label="Buffer-Based")
plt.legend()
plt.title("Selected Bitrate")

plt.subplot(3,1,3)
plt.plot([h["buffer"] for h in hist_rate], label="Rate-Based")
plt.plot([h["buffer"] for h in hist_buffer], label="Buffer-Based")
plt.legend()
plt.title("Buffer Level")

plt.tight_layout()
plt.show()
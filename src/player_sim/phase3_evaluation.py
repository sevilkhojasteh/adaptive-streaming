import numpy as np
from abr_simulator import ABRSimulator
from metrics import QoEModel
from trace_loader import load_random_trace

bitrates = [400, 800, 1500, 3000, 6000]

# Load real trace
network_trace = load_random_trace("data/network_traces")
network_trace = network_trace[:100]

# -----------------------------
# Rate-Based
# -----------------------------
sim_rate = ABRSimulator(bitrates)
history_rate = sim_rate.simulate(network_trace, mode="rate")

# -----------------------------
# Buffer-Based
# -----------------------------
sim_buffer = ABRSimulator(bitrates)
history_buffer = sim_buffer.simulate(network_trace, mode="buffer")

# -----------------------------
# QoE Calculation
# -----------------------------
qoe_model = QoEModel()

qoe_rate = qoe_model.compute(
    history_rate,
    sim_rate.rebuffer_time,
    sim_rate.switches
)

qoe_buffer = qoe_model.compute(
    history_buffer,
    sim_buffer.rebuffer_time,
    sim_buffer.switches
)
print("=== Phase 3 Evaluation ===")
print("Rate-Based QoE:", qoe_rate)
print("Buffer-Based QoE:", qoe_buffer)
print("Rate-Based Rebuffer:", sim_rate.rebuffer_time)
print("Buffer-Based Rebuffer:", sim_buffer.rebuffer_time)
print("Rate-Based Switches:", sim_rate.switches)
print("Buffer-Based Switches:", sim_buffer.switches)
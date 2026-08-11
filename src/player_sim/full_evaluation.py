import os
import numpy as np
import matplotlib.pyplot as plt

from streaming_env import StreamingEnv
from rl_abr import RL_ABR
from trace_loader import load_trace
from abr_simulator import ABRSimulator
from metrics import QoEModel

TRACE_FOLDER = "data/network_traces"

bitrates = [400, 800, 1500, 3000, 6000]

qoe_model = QoEModel()

def evaluate_trace(trace):
    # Rate-based
    sim1 = ABRSimulator(bitrates)
    history1 = sim1.simulate(trace)
    qoe1 = qoe_model.compute(history1, sim1.rebuffer_time)

    # Buffer-based
    sim2 = ABRSimulator(bitrates)
    sim2.rate_based_decision = lambda x: sim2.buffer_based_decision()
    history2 = sim2.simulate(trace)
    qoe2 = qoe_model.compute(history2, sim2.rebuffer_time)

    # RL-based
    env = StreamingEnv(trace, bitrates)
    agent = RL_ABR(3, len(bitrates))

    state = env.reset()
    total_reward = 0

    while True:
        action = agent.select_action(state, epsilon=0)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break

    return qoe1, qoe2, total_reward


def main():
    results = []

    files = os.listdir(TRACE_FOLDER)

    for f in files[:10]:  # use first 10 traces
        trace = load_trace(os.path.join(TRACE_FOLDER, f))
        trace = trace[:200]

        q1, q2, q3 = evaluate_trace(trace)
        results.append([q1, q2, q3])

        print(f"{f} → Rate:{q1:.2f}, Buffer:{q2:.2f}, RL:{q3:.2f}")

    results = np.array(results)

    print("\nAverage QoE:")
    print("Rate-Based:", np.mean(results[:,0]))
    print("Buffer-Based:", np.mean(results[:,1]))
    print("RL-Based:", np.mean(results[:,2]))

    # Plot comparison
    plt.boxplot(results, labels=["Rate", "Buffer", "RL"])
    plt.title("QoE Comparison Across Traces")
    plt.ylabel("QoE Score")
    plt.show()


if __name__ == "__main__":
    main()

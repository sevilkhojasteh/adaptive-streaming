"""
Phase 5 — Full evaluation of Rate-Based, Buffer-Based, and RL ABR agents.
Runs each algorithm on N traces, computes QoE, and produces plots.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from streaming_env import StreamingEnv
from abr_simulator import ABRSimulator
from rl_abr import RL_ABR
from trace_loader import load_random_trace
from metrics import QoEModel

# -------- CONFIG --------
BITRATES     = [400, 800, 1500, 3000, 6000]
TRACE_DIR    = "data/network_traces"
MODEL_PATH   = "models/rl_abr_ep600.pt"        # change if your latest ckpt differs
NUM_TRACES   = 20
TRACE_LEN    = 200
RESULTS_DIR  = "results"
# ------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)
qoe_model = QoEModel(alpha=4.3, beta=1.0, mode="log")


def evaluate_rl(trace, agent):
    env = StreamingEnv(trace, BITRATES)
    state = env.reset()
    history = []
    while True:
        # deterministic: force ε=0 for evaluation
        agent.epsilon = 0.0
        action = agent.select_action(state)
        state, _, done = env.step(action)
        history.append({"bitrate": BITRATES[action], "buffer": env.buffer})
        if done:
            break
    return history, env.rebuffer_time


def evaluate_baseline(trace, mode):
    sim = ABRSimulator(BITRATES)
    sim.simulate(trace, mode=mode)
    return sim.history, sim.rebuffer_time


def main():
    # --- Load trained RL agent ---
    print(f"Loading RL agent from {MODEL_PATH} ...")
    agent = RL_ABR(state_dim=3, action_dim=len(BITRATES))
    agent.load(MODEL_PATH)

    results = {"Rate-Based": [], "Buffer-Based": [], "RL (DQN)": []}
    rebuffers = {k: [] for k in results}
    bitrates_used = {k: [] for k in results}
    switch_counts = {k: [] for k in results}

    print(f"\nEvaluating on {NUM_TRACES} random traces of length {TRACE_LEN} ...\n")

    for i in range(NUM_TRACES):
        trace = load_random_trace(TRACE_DIR)[:TRACE_LEN]

        for name, runner in [
            ("Rate-Based",   lambda t: evaluate_baseline(t, "rate")),
            ("Buffer-Based", lambda t: evaluate_baseline(t, "buffer")),
            ("RL (DQN)",     lambda t: evaluate_rl(t, agent)),
        ]:
            history, rebuf = runner(trace)
            qoe = qoe_model.compute(history, rebuf)
            brs = [h["bitrate"] for h in history]
            switches = sum(1 for j in range(1, len(brs)) if brs[j] != brs[j-1])

            results[name].append(qoe)
            rebuffers[name].append(rebuf)
            bitrates_used[name].append(np.mean(brs))
            switch_counts[name].append(switches)

        print(f"[{i+1:>2}/{NUM_TRACES}] "
              f"RB={results['Rate-Based'][-1]:8.2f}  "
              f"BB={results['Buffer-Based'][-1]:8.2f}  "
              f"RL={results['RL (DQN)'][-1]:8.2f}")

    # -------- Summary table --------
    print("\n" + "=" * 78)
    print(f"{'Algorithm':<15}{'Avg QoE':>12}{'Avg Bitrate':>14}"
          f"{'Rebuffer (s)':>15}{'Switches':>12}")
    print("-" * 78)
    for name in results:
        print(f"{name:<15}"
              f"{np.mean(results[name]):>12.2f}"
              f"{np.mean(bitrates_used[name]):>14.0f}"
              f"{np.mean(rebuffers[name]):>15.2f}"
              f"{np.mean(switch_counts[name]):>12.1f}")
    print("=" * 78)

    # -------- Plots --------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1. QoE boxplot
    axes[0, 0].boxplot(list(results.values()), labels=list(results.keys()))
    axes[0, 0].set_title("QoE Distribution (higher is better)")
    axes[0, 0].set_ylabel("QoE Score")
    axes[0, 0].grid(alpha=0.3)

    # 2. Avg bitrate
    axes[0, 1].bar(bitrates_used.keys(),
                   [np.mean(v) for v in bitrates_used.values()],
                   color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0, 1].set_title("Average Bitrate Chosen")
    axes[0, 1].set_ylabel("kbps")
    axes[0, 1].grid(alpha=0.3, axis="y")

    # 3. Rebuffering
    axes[1, 0].bar(rebuffers.keys(),
                   [np.mean(v) for v in rebuffers.values()],
                   color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1, 0].set_title("Average Rebuffering (lower is better)")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].grid(alpha=0.3, axis="y")

    # 4. Switches
    axes[1, 1].bar(switch_counts.keys(),
                   [np.mean(v) for v in switch_counts.values()],
                   color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1, 1].set_title("Bitrate Switches (lower = smoother)")
    axes[1, 1].set_ylabel("Number of switches")
    axes[1, 1].grid(alpha=0.3, axis="y")

    plt.suptitle("Adaptive Bitrate Streaming — Algorithm Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "abr_comparison.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\n✅ Saved figure → {out_path}")
    plt.show()

    # -------- Save raw results for reproducibility --------
    np.savez(os.path.join(RESULTS_DIR, "eval_results.npz"),
             qoe=results, rebuffers=rebuffers,
             bitrates=bitrates_used, switches=switch_counts)
    print(f"✅ Saved raw results → {RESULTS_DIR}/eval_results.npz")


if __name__ == "__main__":
    main()
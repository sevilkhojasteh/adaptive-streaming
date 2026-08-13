import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("results/reward_history.npy")
window = 20
smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(rewards, alpha=0.3, color="steelblue", label="Per-episode reward")
ax.plot(range(window-1, len(rewards)), smoothed, color="darkblue",
        linewidth=2, label=f"Moving average (window={window})")
ax.axhline(0, color="k", linestyle="--", alpha=0.5)
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.set_title("DQN ABR Agent — Training Progress")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/training_curve.png", dpi=140)
plt.show()
print("✅ Saved → results/training_curve.png")
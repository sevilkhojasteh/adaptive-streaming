from streaming_env import StreamingEnv
from rl_abr import RL_ABR
from trace_loader import load_random_trace
import numpy as np
import os

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

bitrates = [400, 800, 1500, 3000, 6000]
trace = load_random_trace("data/network_traces")[:200]

env = StreamingEnv(trace, bitrates)
agent = RL_ABR(state_dim=3, action_dim=len(bitrates))

episodes = 600
reward_history = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0.0
    step = 0

    # ---- inner loop: one episode ----
    while True:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        agent.store((state, action, reward, next_state, float(done)))

        if step % 4 == 0:
            agent.train()

        state = next_state
        total_reward += reward
        step += 1
        if done:
            break
    # ---- end inner loop ----

    agent.decay_epsilon()
    reward_history.append(total_reward)
    avg = np.mean(reward_history[-20:])
    print(f"Ep {ep:3d} | Reward: {total_reward:8.2f} | Avg20: {avg:8.2f} | ε: {agent.epsilon:.3f}")

    # ---- save checkpoint every 50 episodes ----
    if (ep + 1) % 50 == 0 or ep == episodes - 1:
        agent.save(f"models/rl_abr_ep{ep+1}.pt")
        print(f"  💾 Saved model at episode {ep+1}")

# ---- after ALL episodes finish ----
np.save("results/reward_history.npy", np.array(reward_history))
print("\n✅ Training complete. Reward history saved to results/reward_history.npy")
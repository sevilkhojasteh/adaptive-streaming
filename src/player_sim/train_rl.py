from streaming_env import StreamingEnv
from rl_abr import RL_ABR
from trace_loader import load_random_trace
import numpy as np

bitrates = [400, 800, 1500, 3000, 6000]
trace = load_random_trace("data/network_traces")[:200]

env = StreamingEnv(trace, bitrates)
agent = RL_ABR(state_dim=3, action_dim=len(bitrates))

episodes = 300
reward_history = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0.0
    step = 0

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

    agent.decay_epsilon()
    reward_history.append(total_reward)
    avg = np.mean(reward_history[-20:])
    print(f"Ep {ep:3d} | Reward: {total_reward:8.2f} | Avg20: {avg:8.2f} | ε: {agent.epsilon:.3f}")
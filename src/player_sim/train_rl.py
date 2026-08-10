from streaming_env import StreamingEnv
from rl_abr import RL_ABR
from trace_loader import load_random_trace
import numpy as np

bitrates = [400, 800, 1500, 3000, 6000]

trace = load_random_trace("data/network_traces")
trace = trace[:200]

env = StreamingEnv(trace, bitrates)

agent = RL_ABR(state_dim=3, action_dim=len(bitrates))

episodes = 20

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        agent.store((state, action, reward, next_state, done))
        agent.train()

        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {ep}, Total Reward: {total_reward}")

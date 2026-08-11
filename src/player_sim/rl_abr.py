import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class RL_ABR:
    def __init__(self, state_dim, action_dim, lr=5e-4):
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())   # ✅ sync

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()   # ✅ Huber loss, more stable than MSE

        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.action_dim = action_dim

        # ✅ epsilon-greedy schedule
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        # ✅ how often to sync target net
        self.update_target_every = 500
        self.train_step = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state = torch.FloatTensor(np.array(state)).unsqueeze(0)
            q_values = self.model(state)
        return int(torch.argmax(q_values, dim=1).item())

    def store(self, transition):
        self.memory.append(transition)

    def train(self, batch_size=64):
        if len(self.memory) < 1000:      # ✅ warm-up
            return 0.0

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # ✅ convert via np.array first (fixes the warning + is faster)
        states      = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions     = torch.LongTensor(actions).unsqueeze(1)
        rewards     = torch.FloatTensor(rewards)
        dones       = torch.FloatTensor(dones)

        # Q(s,a)
        q_values = self.model(states).gather(1, actions).squeeze(1)

        # target: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        # ✅ clip gradients to avoid explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ✅ periodically sync target network
        self.train_step += 1
        if self.train_step % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        torch.save({
            'model':        self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'optimizer':    self.optimizer.state_dict(),
            'epsilon':      self.epsilon,
            'train_step':   self.train_step,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        self.target_model.load_state_dict(ckpt['target_model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon    = ckpt['epsilon']
        self.train_step = ckpt.get('train_step', 0)
        self.model.eval()
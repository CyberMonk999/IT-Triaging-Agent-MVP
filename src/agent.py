# src/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.qnetwork import QNetwork

class TicketAgent:
    def __init__(self, input_dim, num_teams, gamma=0.9, epsilon=0.1, lr=1e-3):
        """
        Reinforcement learning agent for IT ticket assignment.
        
        Args:
            input_dim (int): Size of ticket embeddings
            num_teams (int): Number of teams / actions
            gamma (float): Discount factor
            epsilon (float): Exploration rate
            lr (float): Learning rate
        """
        self.q_net = QNetwork(input_dim, num_teams)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_teams = num_teams

    def select_action(self, state_tensor):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_teams)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values).item())

    def train_step(self, state_tensor, action, reward, next_state_tensor):
        """Performs one Q-learning update step."""
        self.q_net.train()
        q_values = self.q_net(state_tensor)
        with torch.no_grad():
            target_q = reward + self.gamma * torch.max(self.q_net(next_state_tensor))
        loss = self.criterion(q_values[action], target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

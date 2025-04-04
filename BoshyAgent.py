import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import tensor

from QNet import QNet

action_map = ["Stop", "Jump", "Left", "JumpLeft", "Right", "JumpRight", "Shoot", "JumpShoot", "LeftShoot",
              "JumpLeftShoot", "RightShoot", "JumpRightShoot"]


class BoshyAgent:
    def __init__(self, settings):
        required_settings = ["gamma", "lr", "batch_size", "n_actions", "mem_size", "input_dims"]
        for setting in required_settings:
            if setting not in settings:
                raise ValueError(f"Missing setting: {setting}")
        self.gamma = settings['gamma']
        self.lr = settings["lr"]
        self.batch_size = settings["batch_size"]
        self.mem_size = settings["mem_size"]
        self.exploration_factor = settings["exploration_factor"]
        input_dims = settings["input_dims"]
        n_actions = settings["n_actions"]

        self.exploration = 0
        self.exploitation = 0
        self.iteration = 1
        self.mem_cntr = 0

        self.main_network = QNet(self.lr, n_actions=n_actions, input_dims=input_dims)
        self.critic_network = QNet(self.lr, n_actions=n_actions, input_dims=input_dims)
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, bool)

    def store_transition(self, state, action, reward, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, observation: np.ndarray, verbose=False):
        state = tensor(observation, dtype=torch.float32).to(self.main_network.device)
        actions = self.main_network.forward(state)
        action_mem = (tensor(self.action_memory[(self.state_memory == observation).all(1)], dtype=torch.float32)
                      .to(self.main_network.device))
        memory_hist = torch.histc(action_mem, bins=actions.size(0), min=0.0, max=12.0) + 1
        ucb = self.exploration_factor * torch.sqrt(np.log(self.iteration) / memory_hist)

        action = torch.argmax(actions + ucb).item()
        self.iteration += 1

        if action == torch.argmax(actions).item():
            self.exploitation += 1
        else:
            self.exploration += 1
        if verbose and self.iteration % 200 == 0:
            print("State:", state)
            print("Histogram", memory_hist)
            print("UCB:", ucb, "\nActions:", actions)
            print("Exploitation:", self.exploitation, "Exploration:", self.exploration)
            print("=================")
            print("Action: ", action, "Max Q:", torch.argmax(actions).item())
        return action

    def learn(self, verbose=False):
        if self.mem_cntr - 1 < self.batch_size:
            return
        device = self.main_network.device
        self.critic_network.optimizer.zero_grad()
        max_mem = min(self.mem_cntr - 1, self.mem_size)
        # e.g. [3, 7, 2, 1, 0]
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        # e.g. [0, 1, 2, 3, 4]
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # e.g. [(0.1, 0.2), (0.04, 0.8), (0.2, 0.9)...]
        state_batch = tensor(self.state_memory[batch]).to(device)
        next_state_batch = tensor(self.state_memory[batch + 1]).to(device)
        # e.g. [0.1, 0.04, 0.2...]
        reward_batch = tensor(self.reward_memory[batch]).to(device)
        # e.g. [0, 9, 13...]
        action_batch = self.action_memory[batch]

        q_eval = self.critic_network.forward(state_batch)[batch_index, action_batch]
        q_target = reward_batch + self.gamma * self.critic_network.forward(next_state_batch).max(dim=1)[0]

        loss = torch.sqrt(self.critic_network.loss(q_target, q_eval).to(device))
        loss.backward()
        self.critic_network.optimizer.step()

    def print_bias(self):
        for seq in self.main_network.children():
            i = 0
            for child in seq.children():
                i += 1
                if i == 5:
                    print("Bias:")
                    print(child.state_dict()['bias'])
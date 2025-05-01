import numpy as np
import torch
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
        self.iteration = 1
        self.mem_cntr = 0
        self.exploration_chosen = 0
        self.exploitation_chosen = 0

        self.q_network = QNet(self.lr, n_actions=n_actions, input_dims=input_dims)
        self.target_network = QNet(self.lr, n_actions=n_actions, input_dims=input_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.target_update_freq = 1000
        self.learn_step_counter = 0

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, done, next_state=None):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        if next_state is not None:
            self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, observation: np.ndarray, verbose=False):
        state = tensor(observation, dtype=torch.float32).to(self.q_network.device)
        actions = self.q_network.forward(state)

        action_mem = (tensor(self.action_memory[(self.state_memory == observation).all(1)], dtype=torch.float32)
                      .to(self.q_network.device))
        memory_hist = torch.histc(action_mem, bins=actions.size(0), min=0.0, max=12.0) + 1
        ucb = self.exploration_factor * torch.sqrt(np.log(self.iteration) / memory_hist)

        action = torch.argmax(actions + ucb).item()

        self.iteration += 1
        if action == torch.argmax(actions).item():
            self.exploitation_chosen += 1
        else:
            self.exploration_chosen += 1
        if self.iteration % 200 == 0:
            print(f"Exploration chosen: {self.exploration_chosen}, Exploitation chosen: {self.exploitation_chosen}")
            print(f"actions: {actions}, ucb: {ucb}")
        return action

    def learn(self, verbose=False):
        if self.mem_cntr < self.batch_size:
            return

        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            if verbose:
                print("Target network updated")

        self.learn_step_counter += 1

        device = self.q_network.device

        max_mem = min(self.mem_cntr, self.mem_size)
        batch_size = min(self.batch_size, max_mem)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        batch_index = np.arange(batch_size, dtype=np.int32)

        state_batch = tensor(self.state_memory[batch]).to(device)
        next_state_batch = tensor(self.next_state_memory[batch]).to(device)
        reward_batch = tensor(self.reward_memory[batch]).to(device)
        terminal_batch = tensor(self.terminal_memory[batch]).to(device)
        action_batch = self.action_memory[batch]

        self.q_network.optimizer.zero_grad()

        q_eval = self.q_network.forward(state_batch)[batch_index, action_batch]
        with torch.no_grad():
            q_next = self.target_network.forward(next_state_batch)
            q_next[terminal_batch] = 0.0  # Set Q-values to 0 for terminal states
            q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.q_network.loss(q_target, q_eval).to(device)

        if self.mem_cntr % 1000 == 0:
            print(f"Loss: {loss.item()}")

        loss.backward()
        self.q_network.optimizer.step()
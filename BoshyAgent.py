import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import random
from torch import tensor

from QNet import QNet

class BoshyAgent:
    def __init__(self, gamma, lr, input_dims, batch_size, n_actions, max_mem_size=100000,
                 graph=False, exploration_factor=0.3, x_grid=1000, y_grid=50):
        self.gamma = gamma
        self.lr = lr
        self.exploration_factor = exploration_factor
        self.graph = graph
        self.mem_size = max_mem_size
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.exploration = 0
        self.exploitation = 0
        self.n_actions = n_actions

        self.batch_size = batch_size
        self.mem_cntr = 0
        self.min_loss = tensor(torch.inf)

        self.main_network = QNet(self.lr, n_actions=n_actions, input_dims=input_dims)
        self.target_network = QNet(self.lr, n_actions=n_actions, input_dims=input_dims)
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.downsampled_state_memory = np.zeros((self.mem_size, 5))
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.downsampled_state_memory[index] = self.downsample(state)
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1
        if self.mem_cntr % 300 == 0:
            print(self.mem_cntr)

    def choose_action(self, observation, iteration, verbose=False):
        state = tensor(observation, dtype=torch.float32).to(self.main_network.device)
        downsampled_state = self.downsample(observation)
        actions = self.main_network.forward(state)
        clipped_cntr = np.min((self.mem_cntr + 1, self.mem_size))
        state_memory = np.all(self.downsampled_state_memory == downsampled_state, axis=1).astype(int)[:clipped_cntr]
        action_mem = tensor(self.action_memory[np.where(state_memory)], dtype=torch.float32).to(self.main_network.device)
        memory_hist = torch.histc(action_mem, bins=actions.size(0), min=-1e-8, max=actions.size(0) + 1e-8) + 1
        ucb = self.exploration_factor * torch.sqrt(np.log(iteration) / memory_hist)

        action = torch.argmax(actions + ucb).item()

        if action == torch.argmax(actions).item():
            self.exploitation += 1
        else:
            self.exploration += 1
        if verbose and iteration % 200 == 0:
            print("State:", state, "Downsampled:", downsampled_state)
            print("Histogram", memory_hist)
            print("UCB:", ucb, "Actions:", actions)
            print("Exploitation:", self.exploitation, "Exploration:", self.exploration)
            print("=================")
            print("Action: ", action, "Max Q:", torch.argmax(actions).item())
        return action

    def learn(self, verbose=False):
        if self.mem_cntr < self.batch_size:
            return
        self.main_network.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = tensor(self.state_memory[batch]).to(self.main_network.device)
        new_state_batch = tensor(self.new_state_memory[batch]).to(self.main_network.device)
        reward_batch = tensor(self.reward_memory[batch]).to(self.main_network.device)
        terminal_batch = tensor(self.terminal_memory[batch]).to(self.main_network.device)
        action_batch = self.action_memory[batch]

        q_eval = self.main_network.forward(state_batch)[batch_index, action_batch]
        q_next = self.target_network.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = torch.sqrt(self.main_network.loss(q_target, q_eval).to(self.main_network.device))
        if loss < self.min_loss:
            self.min_loss = loss
        if verbose:
            print("Loss:", loss.item())
            print(list(q_eval[:1].detach().cpu().numpy()), list(q_target[:1].detach().cpu().numpy()))
        loss.backward()
        self.main_network.optimizer.step()

        if self.graph:
            self.graph_net(64 ** 2 * self.main_network.input_dims)

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def graph_net(self, weights_lcm):
        device = self.main_network.device
        plt.gcf().clear()
        weight_data = tensor([]).to(device)
        bias_data = tensor([]).to(device)
        for layer in [self.main_network.fc1, self.main_network.fc2, self.main_network.fc3, self.main_network.fc4,
                      self.main_network.fc5, self.main_network.fc6, self.main_network.fc7]:
            w = layer.weight.clone().detach()
            b = layer.bias.clone().detach()
            weights = tensor([]).to(device)
            for inputs in range(w.shape[1]):
                input_weights = w[:, inputs].repeat_interleave(weights_lcm // w.nelement())
                weights = torch.cat((weights, input_weights))
            weight_data = torch.cat((weight_data, weights.unsqueeze(0)))
            b = b.repeat_interleave(weights_lcm // b.nelement())
            bias_data = torch.cat((bias_data, b.unsqueeze(0)))
        weight_data = weight_data.cpu().numpy().transpose()
        bias_data = bias_data.cpu().numpy().transpose()
        plt.gcf().subplots(1, 2)
        plt.gcf().axes[0].imshow(weight_data, cmap='viridis', interpolation='none')
        plt.gcf().axes[0].set_aspect("auto")
        plt.gcf().axes[1].imshow(bias_data, interpolation='none')
        plt.gcf().axes[1].set_aspect("auto")
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.flush_events()

    def downsample(self, state):
        return np.array((
            int(state[0] * self.x_grid),
            int(state[1] * self.y_grid),
            int(state[4]),
            int(state[5]),
            int(state[6])
        ), dtype=int)

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import tensor

from QNet import QNet


class BoshyAgent:
    def __init__(self, settings):
        required_settings = ["gamma", "lr", "batch_size", "n_actions", "mem_size", "horizon", "input_dims",
                             "graph", "x_grid", "y_grid"]
        for setting in required_settings:
            if setting not in settings:
                raise ValueError(f"Missing setting: {setting}")
        self.gamma = settings['gamma']
        self.lr = settings["lr"]
        self.batch_size = settings["batch_size"]
        self.mem_size = settings["mem_size"]
        self.horizon = settings["horizon"]
        self.graph = settings["graph"]
        self.exploration_factor = settings["exploration_factor"]
        self.x_grid = settings["x_grid"]
        self.y_grid = settings["y_grid"]
        input_dims = settings["input_dims"]
        n_actions = settings["n_actions"]

        self.exploration = 0
        self.exploitation = 0
        self.iteration = 0
        self.mem_cntr = 0
        self._min_loss = tensor(torch.inf)

        self.main_network = QNet(self.lr, n_actions=n_actions, input_dims=input_dims)
        self.target_network = QNet(self.lr, n_actions=n_actions, input_dims=input_dims)
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.downsampled_state_memory = np.zeros((self.mem_size, 2))
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, bool)

    @property
    def min_loss(self):
        return self._min_loss

    @min_loss.setter
    def min_loss(self, value):
        self._min_loss = value

    def store_transition(self, state, action, reward, done):
        index = self.mem_cntr % (self.mem_size - 1)
        self.state_memory[index] = state
        self.downsampled_state_memory[index] = self.downsample(state)
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1
        if self.mem_cntr % 300 == 0:
            print(self.mem_cntr)

    def choose_action(self, observation, verbose=False):
        state = tensor(observation, dtype=torch.float32).to(self.main_network.device)
        downsampled_state = self.downsample(observation)
        actions = self.main_network.forward(state)
        clipped_cntr = np.min((self.mem_cntr + 1, self.mem_size))
        state_memory = np.all(self.downsampled_state_memory == downsampled_state, axis=1).astype(int)[:clipped_cntr]
        action_mem = (tensor(self.action_memory[np.where(state_memory)], dtype=torch.float32)
                      .to(self.main_network.device))
        memory_hist = torch.histc(action_mem, bins=actions.size(0), min=-1e-8, max=actions.size(0) + 1e-8) + 1
        ucb = self.exploration_factor * torch.sqrt(np.log(self.iteration) / memory_hist)

        action = torch.argmax(actions + ucb).item()
        self.iteration += 1

        if action == torch.argmax(actions).item():
            self.exploitation += 1
        else:
            self.exploration += 1
        if verbose and self.iteration % 200 == 0:
            print("State:", state, "Downsampled:", downsampled_state)
            print("Histogram", memory_hist)
            print("UCB:", ucb, "Actions:", actions)
            print("Exploitation:", self.exploitation, "Exploration:", self.exploration)
            print("=================")
            print("Action: ", action, "Max Q:", torch.argmax(actions).item())
        return action

    def learn_monte_carlo(self, verbose=False):
        if self.mem_cntr < self.batch_size + self.horizon:
            return
        self.main_network.optimizer.zero_grad()
        max_mem = min(self.mem_cntr - self.horizon, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        horizon_indices = np.expand_dims(np.arange(self.horizon), axis=0).repeat(self.batch_size, axis=0)
        new_indices = np.expand_dims(batch, axis=0).repeat(self.horizon, axis=0).transpose() + horizon_indices
        state_batch = tensor(self.state_memory[batch]).to(self.main_network.device)
        # new_state_batch = tensor(self.state_memory[new_indices]).to(self.main_network.device)
        reward_batch = tensor(self.reward_memory[new_indices]).to(self.main_network.device)
        # terminal_batch = tensor(self.terminal_memory[new_indices]).to(self.main_network.device)
        action_batch = self.action_memory[batch]

        q_eval = self.main_network.forward(state_batch)[batch_index, action_batch]
        # q_next = self.target_network.forward(new_state_batch)
        # q_next[terminal_batch] = 0.0
        discounts = tensor(np.expand_dims(np.geomspace(1, self.gamma**(self.horizon - 1), num=self.horizon), axis=0)
                           .repeat(self.batch_size, axis=0), dtype=torch.float32).to(self.main_network.device)
        q_target = (reward_batch * discounts).sum(dim=1)

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
        with torch.no_grad():
            for target_param, main_param in zip(self.target_network.parameters(), self.main_network.parameters()):
                target_param.data *= 0.9
                target_param.data += 0.1 * main_param

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
            int(state[1] * self.y_grid)
        ), dtype=int)

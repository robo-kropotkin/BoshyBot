import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import tensor, save, load
from os.path import exists
import os

from Networks import ActorNetwork, CriticNetwork

action_map = ["Stop", "Jump", "Left", "JumpLeft", "Right", "JumpRight", "Shoot", "JumpShoot", "LeftShoot",
              "JumpLeftShoot", "RightShoot", "JumpRightShoot"]


class BoshyAgent:
    def __init__(self, settings):
        if exists("states.log"):
            os.remove("states.log")
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
        self.n_actions = settings["n_actions"]
        self.mem_cntr = 0

        self.actor_network = ActorNetwork(self.lr, n_actions=self.n_actions, input_dims=input_dims)
        self.critic_network = CriticNetwork(self.lr, input_dims=input_dims)
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, bool)
        self.step = 0
        self.iteration = 1

    def store_transition(self, state, action, reward, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        with open("states.log", "a") as f:
            f.write(f"{state[0]} {state[1]}\n")
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.iteration += 1
        self.mem_cntr += 1

    def choose_action(self, observation: np.ndarray):
        state = tensor(observation, dtype=torch.float32).to(self.actor_network.device)
        dist = self.actor_network.forward(state)
        dist = torch.distributions.Categorical(dist)
        action = dist.sample().item()
        return action

    def save_model(self):
        print("... saving checkpoint ...")
        save(self.actor_network.state_dict(), self.actor_network.checkpoint_file)

    def load_model(self):
        print("... loading checkpoint ...")
        if exists(self.actor_network.checkpoint_file):
            self.actor_network.load_state_dict(load(self.actor_network.checkpoint_file))
            print(self.actor_network.state_dict())

    def get_learn_batch(self):
        max_mem = min(self.mem_cntr - 1, self.mem_size - 1)
        if self.mem_cntr - 1 < self.batch_size:
            batch_size = self.mem_cntr - 1
        else:
            batch_size = self.batch_size
        batch = np.random.choice(max_mem, batch_size, replace=False)
        batch_index = np.arange(batch_size, dtype=np.int32)
        device = self.actor_network.device
        state_batch = tensor(self.state_memory[batch], dtype=torch.float32).to(device)
        next_state_batch = tensor(self.state_memory[batch + 1], dtype=torch.float32).to(device)
        action_batch = tensor(self.action_memory[batch], dtype=torch.int64).to(device)
        reward_batch = tensor(self.reward_memory[batch], dtype=torch.float32).to(device)
        return state_batch, next_state_batch, action_batch, reward_batch, batch_index

    def learn(self):
        self.step += 1
        self.actor_network.optimizer.zero_grad()
        state_batch, next_state_batch, action_batch, reward_batch, batch_index = self.get_learn_batch()
        all_probs = self.actor_network.forward(state_batch) + 1e-10
        selected_probs = all_probs[batch_index, action_batch]
        entropy = -torch.sum(all_probs * torch.log(all_probs), dim=1)
        advantage = (reward_batch + self.gamma * self.critic_network.forward(next_state_batch)
                     - self.critic_network.forward(state_batch))
        actor_loss = (-torch.log(selected_probs) * advantage).mean() - 0.05 * entropy.mean()
        if self.step % 40 == 0:
            print("Actor loss: " + str(actor_loss.item()))
        actor_loss.backward()
        self.actor_network.optimizer.step()
        self.critic_network.optimizer.zero_grad()
        state_batch, next_state_batch, action_batch, reward_batch, batch_index = self.get_learn_batch()
        advantage = (reward_batch + self.gamma * self.critic_network.forward(next_state_batch)
                     - self.critic_network.forward(state_batch))
        critic_loss = advantage.pow(2).mean()
        if self.step % 40 == 0:
            print("Critic loss: " + str(critic_loss.item()))
        critic_loss.backward()
        self.critic_network.optimizer.step()

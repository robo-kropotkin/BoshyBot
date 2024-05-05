# Boshy Bot
import random

from ReadWriteMemory import ReadWriteMemory, ReadWriteMemoryError
from torch import nn, optim, tensor
from gymnasium import Env
from os import getcwd, path

from matplotlib import pyplot as plt
from torch.nn import functional as f
from datetime import datetime, timedelta
from time import sleep

import pygetwindow as gw
import numpy as np

import subprocess
import pyautogui
import keyboard
import torch
import time

MAX_X = 4000
MAX_Y = 500
GRID_SIZE = 50
torch.set_printoptions(precision=2)

class BoshyAgent:
    def __init__(self, gamma, lr, input_dims, batch_size, n_actions, max_mem_size=100000,
                 gain=(1, 1, 1), graph=False, exploration_factor=0.3):
        self.gamma = gamma
        self.lr = lr
        self.exploration_factor = exploration_factor
        self.graph = graph
        self.mem_size = max_mem_size

        self.batch_size = batch_size
        self.mem_cntr = 0
        self.min_loss = tensor(torch.inf)

        self.Q_eval = QNet(self.lr, n_actions=n_actions, input_dims=input_dims)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.downsampled_state_memory = np.zeros((self.mem_size, 2))
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, bool)

    def store_transition(self, state, action, reward, state_, done, verbose=False):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.downsampled_state_memory[index] = np.array((int(state[0] * GRID_SIZE),
                                                         int(state[1] * GRID_SIZE)), dtype=np.int32)
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, observation, iteration, verbose=False):
        if iteration < self.batch_size:
            return random.randint(0, 15)
        state = tensor(observation, dtype=torch.float32).to(self.Q_eval.device)
        downsampled_state = np.array((int(observation[0] * GRID_SIZE),
                                      int(observation[1] * GRID_SIZE)), dtype=int)
        actions = self.Q_eval.forward(state)
        clipped_cntr = np.max((self.mem_cntr + 1, self.mem_size))
        state_memory = (np.sum(self.downsampled_state_memory == downsampled_state, axis=1) == 2)[:clipped_cntr]
        action_mem = tensor(self.action_memory[state_memory], dtype=torch.float32).to(self.Q_eval.device)
        memory_hist = torch.histc(action_mem, bins=actions.size(0), min=-1e-8, max=actions.size(0) + 1e-8)
        ucb = self.exploration_factor * torch.sqrt(np.log(iteration) / memory_hist)

        if verbose:
            data = torch.cat((ucb.unsqueeze(0).transpose(0, 1), actions.unsqueeze(0).transpose(0, 1)), dim=1)
            print("State:", state, "Downsampled:", downsampled_state)
            print("Action mem:", action_mem, "Mem hist:", memory_hist)
            print("UCB | Actions")
            print(data)
            print("=================")
        action = torch.argmax(actions + ucb).item()
        return action

    def learn(self, verbose=False):
        if self.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        # # Calculate mean and standard deviation of Q-values
        # q_mean = q_eval.mean()
        # q_std = q_eval.std()
        # t_mean = q_target.mean()
        # t_std = q_target.std()
        #
        # # Identify outliers (samples more than 3 standard deviations away from the mean)
        # eval_outliers_mask = torch.abs(q_eval - q_mean) > 3 * q_std
        # target_outliers_mask = torch.abs(q_target - t_mean) > 3 * t_std
        # outlier_mask = eval_outliers_mask | target_outliers_mask
        #
        # q_target[outlier_mask] = 0.0
        # q_eval[outlier_mask] = 0.0

        loss = torch.sqrt(self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device))
        if loss < self.min_loss:
            self.min_loss = loss
        if verbose:
            print("Loss:", loss.item())
        loss.backward()
        self.Q_eval.optimizer.step()

        if self.graph:
            graph_net(self.Q_eval, 1024 ** 2 * 5)

class QNet(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, n_actions)
        self.init_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0")
        self.to(self.device)

    def forward(self, x):
        x = f.leaky_relu(self.fc1(x))
        x = f.leaky_relu(self.fc2(x))
        return self.fc3(x)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

class BoshyEnv(Env):
    def __init__(self):
        self.rwm = ReadWriteMemory()
        self.process = None
        self.process_handle = None
        self.im_address = 0
        self.x_pointer = 0
        self.y_pointer = 0
        self.x = 0.0
        self.y = 0.0
        self.j_streak = 1e-8
        self.steps = 0
        self.exploration_bonus = np.array([])

    def reset(self, seed=None, options=None):
        keyboard.release("z")
        keyboard.release("left")
        keyboard.release("right")
        keyboard.release("x")
        keyboard.press("r")
        time.sleep(0.03)
        keyboard.release("r")
        self.j_streak = 0
        self.steps = 0

        keyboard.press("right")
        countdown = 7
        for i in range(0, countdown):
            print(countdown - i)
            sleep(1)
        keyboard.release("right")
        self.read_process()
        return np.array((self.x, self.y, 0, 0, 0, 0))

    def run(self):
        self.process_handle = subprocess.Popen(path.join(getcwd(), r"..\IWBTB\I Wanna Be The Boshy.exe"))
        time.sleep(5)
        win = gw.getWindowsWithTitle("I Wanna Be The Boshy")[0]
        win.move(1000, 0)
        win.maximize()
        try:
            self.start_new_save()

        except NameError as ee:
            print(ee)

    def read_process(self):
        self.process = self.rwm.get_process_by_name("I Wanna Be The Boshy.exe")
        self.process.open()
        self.im_address = 0x400000
        x_offsets = [0x8D0, 0x30, 0x4C]
        y_offsets = [0x3FC, 0x28, 0, 0x7E8, 0x8D0, 0x80, 0x54]
        self.x_pointer = self.process.get_pointer(self.im_address + 0x00059A98, x_offsets)
        self.y_pointer = self.process.get_pointer(self.im_address + 0x00059A1C, y_offsets)
        self.x = self.process.read(self.x_pointer) / MAX_X
        self.y = self.process.read(self.y_pointer) / MAX_Y
        self.process.close()

    @staticmethod
    def start_new_save():
        keyboard.press("enter")
        time.sleep(0.02)
        keyboard.release("enter")
        time.sleep(0.02)
        keyboard.press("enter")
        time.sleep(0.02)
        keyboard.release("enter")
        time.sleep(0.02)
        keyboard.press("delete")
        time.sleep(0.02)
        keyboard.release("delete")
        time.sleep(0.02)
        keyboard.press("enter")
        time.sleep(0.02)
        keyboard.release("enter")
        time.sleep(0.02)
        keyboard.press("down")
        time.sleep(0.02)
        keyboard.release("down")
        time.sleep(0.02)
        keyboard.press("enter")
        time.sleep(0.02)
        keyboard.release("enter")

    def step(self, action, verbose=False):
        self.j_streak += 1
        self.j_streak *= action >> 3 & 1
        old_x = self.x
        old_y = self.y
        self.read_process()
        for i in range(100):
            self.read_process()
            if np.abs(self.x + self.y) < MAX_X + MAX_Y and self.y != 0 and self.y != 8:
                break

        done = (self.y == 0 or self.y == 8)
        delta_x = self.x - old_x
        delta_y = self.y - old_y
        reward = 0
        reward += self.x * 0.9
        reward -= self.y * 0.1
        # reward += self.exploration_bonus[int(self.x * GRID_SIZE), int(self.y * GRID_SIZE), action] * 0.5
        # self.exploration_bonus[int(self.x * GRID_SIZE), int(self.y * GRID_SIZE), action] -= 0.1
        self.steps += 1
        if verbose:
            print("X, Delta X, Delta Y, Reward")
            print(self.x, delta_x, delta_y, reward)
            print("===============")
        observation = np.array((self.x, self.y, delta_x, delta_y, 1 - 1 / (self.j_streak + 1), 1 - 1 / self.steps))
        return observation, reward, done, False, {}

    def render(self):
        pass

    def close(self):
        self.process_handle.terminate()

def graph_net(net, weights_lcm):
    device = net.device
    plt.gcf().clear()
    weight_data = tensor([]).to(device)
    bias_data = tensor([]).to(device)
    for layer in [net.fc1, net.fc2, net.fc3]:
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

def setup_graph():
    plt.ion()
    plt.subplots(2, 1)
    plt.suptitle("Initializing...")
    plt.gcf().canvas.draw_idle()
    plt.gcf().canvas.flush_events()
    plt.pause(0.001)
    win = gw.getWindowsWithTitle("I Wanna Be The Boshy")[0]
    win.activate()

def keyboard_input(action):
    if action & 1:
        keyboard.press("left")
    else:
        keyboard.release("left")
    if action & 2:
        keyboard.press("right")
    else:
        keyboard.release("right")
    if action & 4:
        keyboard.press("z")
    else:
        keyboard.release("z")
    if action & 8:
        keyboard.press("x")
        time.sleep(0.01)
        keyboard.release("x")

if __name__ == "__main__":
    env = BoshyEnv()
    env.run()
    try:
        graph = False
        tuning = False
        if graph:
            setup_graph()
        epochs = 40
        epoch_duration = 300
        losses, x, times_to_goal = [], [], []
        action_delay = 0.125
        action_delay_multiplier = 1
        batch_size = 64
        mem_size = 4096
        lr = 0.03
        exploration_factor = 0.2
        ef_multiplier = 5
        observation = (0, 0, 0, 0, 0, 0)
        agent = BoshyAgent(gamma=0.99, batch_size=batch_size, n_actions=16,
                           input_dims=[len(observation)], lr=lr, gain=(0.1, 1, 1), graph=graph,
                           max_mem_size=mem_size, exploration_factor=exploration_factor)
        for epoch in range(epochs):
            time_to_goal = timedelta.max
            max_x = 0
            try:
                agent.min_loss = tensor(torch.inf)
                if tuning:
                    agent = BoshyAgent(gamma=0.99, batch_size=batch_size, n_actions=16, input_dims=[len(observation)],
                                       lr=lr, gain=(0.1, 1, 1), graph=graph, max_mem_size=1000,
                                       exploration_factor=exploration_factor * ef_multiplier ** epoch)
                iteration = 1
                done = False

                observation = env.reset()
                start_time = datetime.now()

                print("Begin epoch:", epoch + 1)
                if tuning:
                    print("Exploration factor:", exploration_factor * ef_multiplier ** epoch)
                #  while (datetime.now() - start_time).total_seconds() < epoch_duration:  # Use while debugging
                while not done and (datetime.now() - start_time).total_seconds() < epoch_duration:
                    action = agent.choose_action(observation, iteration * (epoch + 1), verbose=False)

                    keyboard_input(action)

                    sleep(action_delay)

                    observation_, reward, done, info, _ = env.step(action, verbose=False)
                    max_x = np.max([observation_[0], max_x])
                    if observation_[1] < 0.7 and time_to_goal == timedelta.max and iteration > batch_size:
                        time_to_goal = datetime.now() - start_time

                    agent.store_transition(observation, action, reward, observation_, done, verbose=False)
                    agent.learn()

                    if iteration == 4 * batch_size and tuning:
                        env.reset()

                    if iteration % batch_size == 0 and True:
                        print("Duration: ", datetime.now() - start_time)
                        agent.learn(verbose=True)
                        print("Reward: ", reward)
                        print("===============")

                    if iteration % 10 == 0:
                        env.read_process()

                    observation = observation_
                    iteration += 1
                agent.learn(verbose=True)
                print("End epoch:", epoch + 1)

            except (ReadWriteMemoryError, KeyboardInterrupt):
                print("Interrupted. How wude.")
                break
            finally:
                current_loss = agent.min_loss.cpu().detach().numpy().item()
                losses.append(current_loss)
                if current_loss < np.min(losses):
                    best_weights = agent.Q_eval.state_dict()
                x.append(max_x)
                times_to_goal.append(time_to_goal)

        plt.ioff()
        plt.gca().clear()
        print("Final Scores:")
        for i in range(len(losses)):
            if tuning:
                print("For exploration factor:", exploration_factor * ef_multiplier ** i)
            print("Lowest loss for epoch", i, " is:", losses[i], "Max X is:", x[i], "Time to subgoal is:", times_to_goal[i])
    except RuntimeError as e:
        raise e
    finally:
        env.close()

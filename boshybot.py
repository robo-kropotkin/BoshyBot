# Boshy Bot
from ReadWriteMemory import ReadWriteMemory, ReadWriteMemoryError
from torch import nn, optim, tensor
from gymnasium import Env, spaces
from os import getcwd, path

from matplotlib import pyplot as plt
from torch.nn import functional as f
from datetime import datetime, timedelta

import pygetwindow as gw
import numpy as np

import subprocess
import pyautogui
import keyboard
import torch
import time

MAX_X = 3000
MAX_Y = 1000
torch.set_printoptions(precision=2)

class BoshyAgent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000,
                 eps_min=0.01, eps_dec=1e-3, gain=(1, 1, 1), graph=False, epsilon_min_actions=700):
        self.gamma = gamma
        self.epsilon = self.eps_max = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.epsilon_min_actions = epsilon_min_actions
        self.epsilon_min_actions_ = 0
        self.lr = lr
        self.actio_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.min_loss = tensor(torch.inf)
        self.graph = graph

        self.Q_eval = QNet(self.lr, n_actions=n_actions, input_dims=input_dims, gain=gain)
        # self.scheduler1 = optim.lr_scheduler.ExponentialLR(self.Q_eval.optimizer1, gamma=0.9)
        # self.scheduler2 = optim.lr_scheduler.ExponentialLR(self.Q_eval.optimizer2, gamma=0.9)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, bool)

    def store_transition(self, state, action, reward, state_, done):
        delta_x = np.abs(state[0] - state_[0])
        delta_y = np.abs(state[1] - state_[1])
        if delta_x > 50 or delta_y > 50:
            return
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = tensor(observation, dtype=torch.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.actio_space)
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

        # Calculate mean and standard deviation of Q-values
        q_mean = q_eval.mean()
        q_std = q_eval.std()
        t_mean = q_target.mean()
        t_std = q_target.std()

        # Identify outliers (samples more than 3 standard deviations away from the mean)
        eval_outliers_mask = torch.abs(q_eval - q_mean) > 3 * q_std
        target_outliers_mask = torch.abs(q_target - t_mean) > 3 * t_std
        outlier_mask = eval_outliers_mask | target_outliers_mask

        q_target[outlier_mask] = 0.0
        q_eval[outlier_mask] = 0.0

        loss = torch.sqrt(self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device))
        if loss < self.min_loss:
            self.min_loss = loss
        if verbose:
            print("Loss:", loss.item())
        loss.backward()
        self.Q_eval.optimizer.step()

        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec

        if self.graph:
            graph_net(self.Q_eval, 1024 ** 2 * 5)

class QNet(nn.Module):
    def __init__(self, lr, input_dims, n_actions, gain):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, n_actions)
        self.fc1.weight = nn.init.xavier_uniform_(self.fc1.weight, gain=gain[0])
        self.fc2.weight = nn.init.xavier_uniform_(self.fc2.weight, gain=gain[0])
        self.fc3.weight = nn.init.xavier_uniform_(self.fc3.weight, gain=gain[0])
        with torch.no_grad():
            for i in range(3, n_actions, 4):
                self.fc3.bias[i] = -1
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.warmup = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.001, end_factor=1, total_iters=1000)
        self.scheduler2 = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)
        self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer,
                                                         schedulers=[self.warmup, self.scheduler2], milestones=[1000])
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0")
        self.to(self.device)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        return self.fc3(x)

class BoshyEnv(Env):
    def __init__(self):
        self.rwm = ReadWriteMemory()
        self.process = None
        self.process_handle = None
        self.im_address = 0
        self.x_pointer = 0
        self.y_pointer = 0
        self.x = 0
        self.y = 0
        self.exploration_bonus = np.array([])
        self.ebd = 3
        self.j_streak = 0
        self.steps = 0

    def reset(self, seed=None, options=None):
        pyautogui.moveTo(3500, 10)
        keyboard.release("z")
        keyboard.release("left")
        keyboard.release("right")
        keyboard.release("x")
        keyboard.press("r")
        time.sleep(0.03)
        keyboard.release("r")
        self.read_process()
        self.exploration_bonus = np.ones((MAX_X // self.ebd, MAX_Y // self.ebd))
        self.j_streak = 0
        self.steps = 0

        keyboard.press("right")
        time.sleep(3.6)
        keyboard.release("right")
        pyautogui.moveTo(3500, 10)
        return self.x, self.y, 0, 0, 0, 0

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
        self.x = self.process.read(self.x_pointer)
        self.y = self.process.read(self.y_pointer)
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

    def step(self, action):
        self.j_streak += 300
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
        reward = self.x * 5 / MAX_X - self.steps / 10
        reward += self.exploration_bonus[self.x // self.ebd, self.y // self.ebd]
        self.exploration_bonus[self.x // self.ebd, self.y // self.ebd] -= 0.1
        self.steps += 1
        return (self.x, self.y, delta_x * 100, delta_y * 100, self.j_streak, self.steps), reward, done, False, {}

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
        epochs = 10
        epoch_duration = 60
        losses, x, times_to_goal = [], [], []
        action_delay = 0.125
        action_delay_multiplier = 1
        epsilon = 1.0
        epsilon_multiplier = np.power(0.2, 1 / 40)
        batch_size = 32
        lr = 0.0003
        observation = env.reset()
        agent = BoshyAgent(gamma=0.99, epsilon=epsilon, batch_size=batch_size, n_actions=16, eps_min=0.0,
                           input_dims=[len(observation)], lr=lr, gain=(0.1, 1, 1), graph=graph, eps_dec=1e-3,
                           max_mem_size=1000)
        for epoch in range(epochs):
            time_to_goal = timedelta.max
            max_x = 0
            try:
                if tuning:
                    agent = BoshyAgent(gamma=0.99, epsilon=epsilon*epsilon_multiplier**epoch, batch_size=batch_size,
                                       n_actions=16, eps_min=0.01, input_dims=[len(observation)], lr=lr, gain=(0.1, 1, 1),
                                       graph=graph, eps_dec=1e-3, max_mem_size=1000)
                iteration = 1
                done = False
                start_time = datetime.now()
                print("Begin epoch:", epoch + 1)
                if tuning:
                    print("Epsilon = ", epsilon*epsilon_multiplier**epoch)
                while not done and (datetime.now() - start_time).total_seconds() < epoch_duration:
                    action = agent.choose_action(observation)
                    if action & 3 == 3:
                        action -= 1

                    keyboard_input(action)

                    start_of_delay = datetime.now()
                    while (datetime.now() - start_of_delay).total_seconds() <\
                            action_delay*action_delay_multiplier**epoch:
                        agent.learn()

                    observation_, reward, done, info, _ = env.step(action)
                    max_x = np.max([observation_[0], max_x])
                    if 1500 > observation_[0] > 1400 and time_to_goal == timedelta.max and iteration > batch_size:
                        time_to_goal = datetime.now() - start_time

                    agent.store_transition(observation, action, reward, observation_, done)

                    if iteration == 4 * batch_size and tuning:
                        env.reset()

                    if iteration % (batch_size * 2) == 0:
                        print("Duration: ", datetime.now() - start_time)
                        agent.learn(verbose=True)
                        print("Reward: ", reward)

                    if iteration % 10 == 0:
                        env.read_process()

                    observation = observation_
                    iteration += 1
                print("End epoch:", epoch + 1)
                observation = env.reset()

            except ReadWriteMemoryError:
                print("Interrupted. How wude.")
                break
            finally:
                losses.append(agent.min_loss.cpu().detach().numpy().item())
                x.append(max_x)
                times_to_goal.append(time_to_goal)
        plt.ioff()
        plt.gca().clear()
        print("Final Scores:")
        for i in range(len(losses)):
            if tuning:
                print("For ", epsilon*epsilon_multiplier**i)
            print("Loss is:", losses[i], "Max X is:", x[i], "Time to subgoal is:", times_to_goal[i])
    except RuntimeError as e:
        raise e
    finally:
        env.close()

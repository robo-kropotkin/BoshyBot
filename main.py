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
import keyboard
import torch
import time

MAX_X = 4000
MAX_Y = 500
X_GRID_SIZE = 1000
Y_GRID_SIZE = 125
torch.set_printoptions(precision=2)

class QNet(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(QNet, self).__init__()
        self.input_dims = input_dims
        self.fc1 = nn.Linear(input_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.dropout1 = nn.Dropout()
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout()
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, n_actions)
        self.init_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0")
        self.to(self.device)

    def forward(self, x):
        x = f.leaky_relu(self.fc1(x), negative_slope=2e-2)
        x = self.dropout1(f.leaky_relu(self.fc2(x), negative_slope=2e-2))
        x = f.leaky_relu(self.fc3(x), negative_slope=2e-2)
        x = f.leaky_relu(self.fc4(x), negative_slope=2e-2)
        x = self.dropout2(f.leaky_relu(self.fc5(x), negative_slope=2e-2))
        x = f.leaky_relu(self.fc6(x), negative_slope=2e-2)
        return self.fc7(x)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.xavier_uniform_(self.fc6.weight)
        nn.init.xavier_uniform_(self.fc7.weight)
        nn.init.constant_(self.fc7.bias, 0)

class BoshyEnv(Env):
    def __init__(self):
        self.rwm = ReadWriteMemory()
        self.process = None
        self.process_handle = None
        self.im_address = self.x_pointer = self.y_pointer = self.j_streak = self.r_streak = self.l_streak = 0
        self.steps = 0
        self.x = self.y = 0.0
        self.exploration_bonus = np.array([])

    def reset(self, seed=None, options=None):
        keyboard.release("z")
        keyboard.release("left")
        keyboard.release("right")
        keyboard.release("x")
        keyboard.press("r")
        time.sleep(0.03)
        keyboard.release("r")
        self.j_streak = self.l_streak = self.r_streak = self.steps = 0

        keyboard.press("right")
        countdown = 2
        for i in range(0, countdown):
            print(countdown - i)
            sleep(1)
        keyboard.release("right")
        self.read_process()
        return np.array((self.x, self.y, 0, 0, 0, 0, 0, 0, 0))

    def get_state(self):
        return np.array((self.x, self.y, 0, 0, self.j_streak / 10, self.l_streak / 100, self.r_streak / 100,
                         self.steps / 100, 0))

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

    def step(self, action, verbose=False, epoch=0, tuning=False):
        self.j_streak += 1
        self.j_streak *= action >> 2 & 1
        self.r_streak += 1
        self.r_streak *= action >> 1 & 1 and not action & 1
        self.l_streak += 1
        self.l_streak *= action & 1 and not action & 2
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
        reward -= 10 * (action >> 3 & 1)
        reward -= 10 * ((action & 3) == 3)
        reward += self.x
        if self.l_streak <= 5 and self.r_streak <= 0:
            reward -= 0.25
        if 0.35 < self.x < 0.475:
            if not tuning:
                curve_distance = (320 * self.x + 50 * self.y - 157) / (10 * np.sqrt(1049))
            else:
                curve_distance = ((270 + 10 * epoch) * self.x + 50 * self.y - 157) / (10 * np.sqrt(1049))
            curve_distance_penalty = curve_distance**4 * 110000
            if self.steps % 200 == 199:
                print(curve_distance)
                print(curve_distance_penalty)
                print(self.x)
            reward -= curve_distance_penalty
        reward = max(min(reward, 100), -100)
        self.steps += 1
        if verbose:
            print("X, Delta X, Delta Y, Reward")
            print(self.x, delta_x, delta_y, reward)
            print("===============")
        observation = np.array((self.x, self.y, delta_x * 10, delta_y * 10, self.j_streak / 10, self.l_streak / 10,
                                self.r_streak / 10, self.steps / 100, action / 15))
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
    for layer in [net.fc1, net.fc2, net.fc3, net.fc4, net.fc5, net.fc6, net.fc7]:
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
        tuning = True
        if graph:
            setup_graph()
        epochs = 10
        epoch_duration = 60
        losses, x, times_to_goal = [], [], []
        action_delay = initial_action_delay = 0.005
        middle_action_delay = 0.001
        action_delay_multiplier = 1
        batch_size = 512
        initial_batch_size = 512
        mem_size = 30000
        lr = 0.003
        exploration_factor = 200
        ef_multiplier = 5
        observation = env.get_state()
        agent = BoshyAgent(gamma=0.99999, initial_batch_size=initial_batch_size, batch_size=batch_size, n_actions=16,
                           input_dims=len(observation), lr=lr, graph=graph,
                           max_mem_size=mem_size, exploration_factor=exploration_factor)
        best_weights = agent.Q_eval.state_dict()
        for epoch in range(epochs):
            max_x = 0
            time_to_goal = timedelta.max
            try:
                print("Begin epoch:", epoch + 1)
                current_loss, max_x, time_to_goal = agent.play(action_delay, epoch)
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
                # agent.Q_eval.init_weights()
                x.append(max_x)
                times_to_goal.append(time_to_goal)
                keyboard_input(0)
                if keyboard.is_pressed("q"):
                    print("Ok, you can quit")
                    break
        # print("Running with best weights...")
        # if not keyboard.is_pressed('w'):
        #     agent.Q_eval.load_state_dict(best_weights)
        #     agent.play()
        if graph:
            plt.ioff()
            plt.gca().clear()
            plt.close()
        print("Final Scores:")
        for i in range(len(losses)):
            print("Lowest loss for epoch", i, " is:", losses[i], "Max X is:", x[i],
                  "Time to subgoal is:", times_to_goal[i])
    except RuntimeError as e:
        raise e
    finally:
        env.close()

# Boshy Bot
from ReadWriteMemory import ReadWriteMemory
from matplotlib.colors import Normalize
from torch import nn, optim, tensor
from gymnasium import Env, spaces
from os import getcwd, path
from tqdm import tqdm

from matplotlib import pyplot as plt
from torch.nn import functional as f

import pygetwindow as gw
import numpy as np

import subprocess
import keyboard
import torch
import time

MAX_X = 3000
MAX_Y = 1000
X_RESOLUTION = 20
Y_RESOLUTION = 20

class BoshyAgent:
    def __init__(self, gamma, epsilon, alpha, input_dims, batch_size, n_actions, max_mem_size=100000,
                 eps_min=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.alpha = alpha
        self.actio_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = QNet(self.alpha, n_actions=n_actions, input_dims=input_dims, fc1_dims=128, fc2_dims=128)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = tensor([observation], dtype=torch.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.actio_space)
        return action

    def learn(self, print_):
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

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        if print_:
            print(loss)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    @staticmethod
    def act(action):
        if action & 1:
            keyboard.press("left")
        else:
            keyboard.release("left")
        if action & 2:
            keyboard.press("right")
        else:
            keyboard.release("right")
        if action & 4:
            keyboard.press("x")
            time.sleep(0.02)
            keyboard.release("x")
        if action & 8:
            keyboard.press('z')
        else:
            keyboard.release('z')


class BoshyEnv(Env):
    def __init__(self):
        self.observation_space = spaces.Dict({
            'x': spaces.Discrete(int(MAX_X / X_RESOLUTION)),
            'y': spaces.Discrete(int(MAX_Y / Y_RESOLUTION)),
            'isDead': spaces.Discrete(2)
        })
        self.action_space = spaces.Discrete(16)
        self.rwm = ReadWriteMemory()
        self.gameOpen = False
        self.isRunnning = False
        self.process = None
        self.process_handle = None
        self.im_address = 0
        self.death_tracker_pointer = 0
        self.x_pointer = 0
        self.y_pointer = 0
        self.x = 0
        self.y = 0

    def run(self):
        self.process_handle = subprocess.Popen(path.join(getcwd(), r"..\IWBTB\I Wanna Be The Boshy.exe"))
        try:
            self.start_new_save()
            self.isRunnning = True
            self.read_process()
            self.x = int(self.process.read(self.x_pointer) / X_RESOLUTION)
            self.y = int(self.process.read(self.y_pointer) / Y_RESOLUTION)

        except NameError as ee:
            print(ee)

    def read_process(self):
        self.process = self.rwm.get_process_by_name("I Wanna Be The Boshy.exe")
        self.process.open()
        self.im_address = 0x400000
        dt_offsets = [0x1D8, 0x18, 0, 0x60, 0x34, 0x3C, 0x238]
        x_offsets = [0x8D0, 0x30, 0x4C]
        y_offsets = [0x3FC, 0x28, 0, 0x7E8, 0x8D0, 0x80, 0x54]
        self.death_tracker_pointer = self.process.get_pointer(self.im_address + 0x00059A94, dt_offsets)
        self.x_pointer = self.process.get_pointer(self.im_address + 0x00059A98, x_offsets)
        self.y_pointer = self.process.get_pointer(self.im_address + 0x00059A1C, y_offsets)

    def print_env(self):
        print("X = " + str(self.process.read(self.x_pointer)))
        print("Y = " + str(self.process.read(self.y_pointer)))

    @staticmethod
    def start_new_save():
        time.sleep(5)
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
        time.sleep(2)

    def step(self, action):
        time.sleep(0.02)
        x = self.process.read(self.x_pointer)
        y = self.process.read(self.y_pointer)
        done = (y == 0 or y == 8)
        penalty_squares = [[1525, 350, 500, 100, 100]]
        penalty = 0
        if not done:
            self.x = x // X_RESOLUTION + ((action & 2) >> 2) - (action & 1)
            self.y = y // Y_RESOLUTION
            for square in penalty_squares:
                if in_square(self.x, self.y, square):
                    penalty += square[4]
                if x > 1300:
                    penalty += self.y * 3
        return (self.x, self.y), self.x * 3 - penalty, done, False, {}

    def reset(self, seed=None, options=None):
        keyboard.release("z")
        keyboard.press("r")
        time.sleep(0.03)
        keyboard.release("r")
        time.sleep(1)
        self.read_process()
        self.x = int(self.process.read(self.x_pointer) / X_RESOLUTION)
        self.y = int(self.process.read(self.y_pointer) / Y_RESOLUTION)
        return np.array([self.x, self.y], dtype=np.float32)

    def render(self):
        pass

    def close(self):
        self.process_handle.terminate()


class QNet(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0")
        self.to(self.device)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        return self.fc3(x)


def in_square(x, y, square):
    result = x >= square[0] / X_RESOLUTION and y >= square[1] / Y_RESOLUTION
    result = result and x <= (square[0] + square[2]) / X_RESOLUTION
    return result and y <= (square[1] + square[3]) / Y_RESOLUTION


def q_learning():
    env = BoshyEnv()
    q_table = np.zeros((MAX_X // X_RESOLUTION, MAX_Y // Y_RESOLUTION, 32))
    prior_knowledge = [[1510, 360, 500, 100, -100], [1450, 250, 200, 70, 100]]

    for i in tqdm(range(MAX_X // X_RESOLUTION), desc="Initializing..."):
        for j in range(MAX_Y // Y_RESOLUTION):
            for k in range(32):
                value = i  # + ((k & 2) >> 1) - (k & 1)  # + random.randint(-2, 2)
                if i * X_RESOLUTION > 1200:
                    value -= 3 * j
                for square in prior_knowledge:
                    if in_square(i, j, square):
                        value += square[4] - i * 2
                q_table[i, j, k] = value
                if k & 3 == 3:
                    q_table[i, j, k] = -10000

    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration rate
    env.run()
    agent = BoshyAgent()
    for episode in range(1):
        state, _ = env.reset()
        done1 = False
        try:
            frame = 0
            # fr = 0
            preferred_action = 2
            while not done1:
                frame += 1
                if frame == 10:
                    env.read_process()
                    frame = 0
                if np.random.rand() < epsilon:
                    action1 = env.action_space.sample()
                else:
                    action1 = np.argmax(q_table[state[0], state[1]])
                    if q_table[state[0], state[1], preferred_action] == q_table[state[0], state[1], action1]:
                        action1 = preferred_action

                if (action1 & 3) == 3:
                    action1 -= 1
                agent.act(action1)
                next_state, reward1, done1, _, _ = env.step(action1)
                next_max = np.max(q_table[next_state[0], next_state[1], :])
                # if frame == 9:
                #     q_table_list = q_table.tolist()
                #     with open("table.json", 'w') as json_file:
                #         json.dump(q_table_list, json_file, indent=4)
                old_value = q_table[state[0], state[1], action1]
                new_value = (1 - alpha) * old_value + alpha * ((1 - gamma) * float(reward1) + gamma * next_max)
                q_table[state[0], state[1], action1] = new_value
                # if new_value > old_value:
                # print(state[0], state[1], old_value, reward1, next_max, action1, np.argmax(q_table[next_state[0],
                # next_state[1]]), new_value)
                # print(q_table[76, 19, 2])

                state = next_state
        except IndexError as e:
            print("E", e)

    fig, axs = plt.subplots(2, 3, figsize=(12, 9))
    vmin = q_table[:, :, 0].min()
    vmax = q_table[:, :, 0].max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    for i in range(3):
        for j in range(2):
            ax = axs[j, i]
            ax.imshow(q_table[:, :, i + j * 8].transpose(), norm=norm)
            ax.set_aspect("auto")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(axs[0, 0].imshow(q_table[:, :, 0].transpose()), cax=cbar_ax)
    axs[0, 0].set_aspect("auto")
    plt.suptitle("2-D Heat Map")
    plt.show()

    env.close()


def deep_q_learning():
    env = BoshyEnv()
    agent = BoshyAgent()
    device = torch.device("cuda")
    q_net = QNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(q_net.parameters(), lr=0.1)

    plt.axis((0, 10, 0, 1))
    env.run()
    for epoch in range(1):
        env.reset()
        time.sleep(1)
        train(q_net, env, agent, device, criterion, optimizer, epoch)

    plt.show()
    env.close()


def train(model, env, agent, device, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    duration = 0
    next_state, reward, done, _, _ = env.step(0)
    plt.ion()
    plt.title("Initializing...")
    plt.gcf().canvas.draw_idle()
    plt.gcf().canvas.flush_events()
    plt.pause(0.001)
    win = gw.getWindowsWithTitle("I Wanna Be The Boshy")[0]
    win.activate()
    targets = tensor(np.zeros(1000) + MAX_X**2).to(device)
    rewards = tensor(np.zeros(1000)).to(device)
    while not done:
        duration += 1
        next_state = torch.tensor(next_state).float().to(device)
        action = int(model(next_state))
        optimizer.zero_grad()
        agent.act(action)
        next_state, reward, done, _, _ = env.step(action)
        reward = torch.tensor([float(reward)], requires_grad=True).to(device)
        rewards = torch.cat((rewards[1:], reward))
        loss = criterion(rewards, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if duration % 200 == 0:
            print(f'Epoch {epoch}, Step {duration}, Loss: {loss / 200:.4f}')
        if duration % 10 == 0:
            plt.gcf().clear()
            data = tensor(np.zeros((1, 256))).to(device)
            for layer in [model.fc1, model.fc2, model.fc3]:
                w = layer.weight.clone().detach()
                b = layer.bias.clone().detach()
                layer_weights = tensor([]).to(device)
                for inputs in range(w.shape[1]):
                    input_weights = w[:, inputs].repeat_interleave(int(256 / w.nelement()))
                    layer_weights = torch.cat((layer_weights, input_weights))
                data = torch.cat((data, layer_weights.unsqueeze(0)))
            data = data.cpu().numpy().transpose()
            plt.rcParams['figure.max_open_warning'] = 0
            plt.gca().imshow(data, cmap='viridis', interpolation='none')  # Change the colormap as needed
            plt.gca().set_aspect("auto")
            plt.title('Random Data Visualization')
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.flush_events()
            env.read_process()
    plt.ioff()


if __name__ == "__main__":
    env = BoshyEnv()
    agent = BoshyAgent(gamma=0.9, epsilon=1.0, batch_size=64, n_actions=16, eps_min=0.01, input_dims=[2], alpha=0.00003)
    scores, eps_history = [], []
    score = 0
    done = False
    env.run()
    observation = env.reset()
    duration = 0
    while not done and duration < 2000:
        action = agent.choose_action(observation)
        agent.act(action)
        observation_, reward, done, info, _ = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        if duration % 100 == 0:
            print(duration)
            agent.learn(True)
        observation = observation_
        duration += 1
    scores.append(score)
    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores[-100:])
    env.close()

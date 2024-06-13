# Boshy Bot
from ReadWriteMemory import ReadWriteMemoryError
from datetime import datetime, timedelta
from torch import tensor
from time import sleep

from matplotlib import pyplot as plt

import pygetwindow as gw
import numpy as np

import keyboard
import torch

from BoshyAgent import BoshyAgent
from BoshyEnv import BoshyEnv

torch.set_printoptions(precision=2)


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
    if action == 0:
        keyboard.press("left")
    else:
        keyboard.release("left")
    if action == 1:
        keyboard.press("right")
    else:
        keyboard.release("right")
    if action == 2:
        keyboard.press("x")
        sleep(0.01)
        keyboard.release("x")
    if action == 3:
        keyboard.press("z")
    if action == 4:
        keyboard.release("z")


class BoshyController:
    def __init__(self, settings=None):
        self.epochs = settings.epochs if hasattr(settings, "epochs") else 10
        self.epoch_duration = settings.epoch_duration if hasattr(settings, "epoch_duration") else 300
        self.action_delay = settings.action_delay if hasattr(settings, "action_delay") else 0.075
        agent_settings = {
            "gamma": settings.gamma if hasattr(settings, "gamma") else 0.99,
            "lr": settings.lr if hasattr(settings, "lr") else 0.03,
            "batch_size": settings.batch_size if hasattr(settings, "batch_size") else 64,
            "n_actions": 5,
            "input_dims": 8,
            "mem_size": settings.mem_size if hasattr(settings, "mem_size") else 100000,
            "horizon": settings.horizon if hasattr(settings, "horizon") else 40,
            "graph": settings.graph if hasattr(settings, "graph") else False,
            "exploration_factor": settings.exploration_factor if hasattr(settings, "exploration_factor") else 0.3,
            "x_grid": settings.x_grid if hasattr(settings, "x_grid") else 20,
            "y_grid": settings.y_grid if hasattr(settings, "y_grid") else 20,
        }
        self.agent = BoshyAgent(agent_settings)
        self.env = BoshyEnv()

    def play(self):
        self.env.run()
        losses, goal_distances = [], []
        for epoch in range(self.epochs):
            self.run_epoch()
            if keyboard.is_pressed("q"):
                break

        self.env.close()
        print("Final Scores:")
        for i in range(len(losses)):
            print("Lowest loss for epoch", i, " is:", losses[i])
            print("Minimal distance to goal for epoch", i, " is:", goal_distances[i])

    def run_epoch(self):
        self.agent.min_loss = tensor(torch.inf)
        self.env.max_x = 0
        iteration = 1
        done = False

        observation = self.env.reset()
        start_time = datetime.now()
        while not done and (datetime.now() - start_time).total_seconds() < self.epoch_duration:
            observation, done = self.step(observation)
            if keyboard.is_pressed("q"):
                break

            if iteration % 10 == 0:
                self.env.read_process()

            iteration += 1

    def step(self, state):
        action = self.agent.choose_action(state, verbose=True)
        keyboard_input(action)

        start_learning = datetime.now()
        while (datetime.now() - start_learning).total_seconds() < self.action_delay:
            self.agent.learn_monte_carlo()
        next_state, reward, done, info, _ = self.env.step(action, verbose=False)
        self.agent.store_transition(state, action, reward, done)
        return next_state, done


if __name__ == "__main__":
    controller = BoshyController()
    controller.play()

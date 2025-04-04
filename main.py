from datetime import datetime
from matplotlib import pyplot as plt

import pygetwindow as gw
import keyboard
import torch

from BoshyAgent import BoshyAgent
from BoshyEnv import BoshyEnv

torch.set_printoptions(precision=2)
action_map = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]
ACTION_DELAY = 0.2
GAMMA = 0.5 ** (ACTION_DELAY / 10)
EF = 0.3


class BoshyController:
    def __init__(self, settings=None):
        self.epochs = settings.epochs if hasattr(settings, "epochs") else 10
        self.epoch_duration = settings.epoch_duration if hasattr(settings, "epoch_duration") else 300
        self.action_delay = settings.action_delay if hasattr(settings, "action_delay") else 0.2
        agent_settings = {
            "gamma": settings.gamma if hasattr(settings, "gamma") else GAMMA,
            "lr": settings.lr if hasattr(settings, "lr") else 0.03,
            "batch_size": settings.batch_size if hasattr(settings, "batch_size") else 64,
            "n_actions": 12,
            "input_dims": BoshyEnv.input_dims(),
            "mem_size": settings.mem_size if hasattr(settings, "mem_size") else 100000,
            "exploration_factor": settings.exploration_factor if hasattr(settings, "exploration_factor") else EF
        }
        self.agent = BoshyAgent(agent_settings)
        self.env = BoshyEnv()

    def play(self):
        self.env.run()
        for epoch in range(self.epochs):
            self.run_epoch()
            if keyboard.is_pressed("q"):
                break

        self.env.close()

    def run_epoch(self):
        self.env.max_x = 0
        done = False

        observation = self.env.reset()
        start_time = datetime.now()
        while not done and (datetime.now() - start_time).total_seconds() < self.epoch_duration:
            observation, done = self.step(observation)
            if keyboard.is_pressed("q"):
                break

    def step(self, state):
        action = self.agent.choose_action(state, verbose=True)
        press = action_map[action]
        if press & 1:
            keyboard.press("z")
        else:
            keyboard.release("z")
        if press & 2:
            keyboard.press("left")
        else:
            keyboard.release("left")
        if press & 4:
            keyboard.press("right")
        else:
            keyboard.release("right")
        if press & 8:
            keyboard.press("x")
        else:
            keyboard.release("x")

        start_learning = datetime.now()
        while (datetime.now() - start_learning).total_seconds() < self.action_delay:
            self.agent.learn()
        next_state, reward, done, info, _ = self.env.step(action, verbose=False)
        self.agent.store_transition(state, action, reward, done)
        return next_state, done

    def close(self):
        self.env.close()


if __name__ == "__main__":
    controller = BoshyController()
    try:
        controller.play()
    except KeyboardInterrupt:
        print("Exiting...")
        controller.close()

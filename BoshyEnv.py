import subprocess
import keyboard
import numpy as np
import pygetwindow as gw
from os import path, getcwd
from time import sleep
from ReadWriteMemory import ReadWriteMemory
from gymnasium import Env

class BoshyEnv(Env):
    def __init__(self, max_x=4000, max_y=500):
        self.rwm = ReadWriteMemory()
        self.process = None
        self.process_handle = None
        self.im_address = self.x_pointer = self.y_pointer = self.j_streak = self.r_streak = self.l_streak = 0
        self.steps = 0
        self.x = self.y = 0.0
        self.max_x = max_x
        self.max_y = max_y
        self.exploration_bonus = np.array([])

    def reset(self, seed=None, options=None):
        keyboard.release("z")
        keyboard.release("left")
        keyboard.release("right")
        keyboard.release("x")
        keyboard.press("r")
        sleep(0.03)
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
        sleep(5)
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
        self.x = self.process.read(self.x_pointer) / self.max_x
        self.y = self.process.read(self.y_pointer) / self.max_y
        self.process.close()

    @staticmethod
    def start_new_save():
        keyboard.press("enter")
        sleep(0.02)
        keyboard.release("enter")
        sleep(0.02)
        keyboard.press("enter")
        sleep(0.02)
        keyboard.release("enter")
        sleep(0.02)
        keyboard.press("delete")
        sleep(0.02)
        keyboard.release("delete")
        sleep(0.02)
        keyboard.press("enter")
        sleep(0.02)
        keyboard.release("enter")
        sleep(0.02)
        keyboard.press("down")
        sleep(0.02)
        keyboard.release("down")
        sleep(0.02)
        keyboard.press("enter")
        sleep(0.02)
        keyboard.release("enter")

    # Epoch is a parameter for tuning environment variables to different values on different epochs
    def step(self, action, verbose=False, epoch=0):
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
            if np.abs(self.x + self.y) < self.max_x + self.max_y and self.y != 0 and self.y != 8:
                break

        done = (self.y == 0 or self.y == 8)
        delta_x = self.x - old_x
        delta_y = self.y - old_y
        reward = 0
        reward -= 10 * (action >> 3 & 1)
        reward -= 10 * ((action & 3) == 3)
        reward += (action >> 1 & 1)
        reward += np.random.normal(scale=0.1)
        reward += self.x
        if self.l_streak <= 25 and self.r_streak <= 25:
            reward -= 0.5
        a = 0.365
        b = 0.43
        if a < self.x < b:
            curve_distance = np.abs(10 * (a - b) * self.y - a + 9 * b - 8 * self.x) \
                             / np.sqrt(25 * (a-b)**2 + 16)
            curve_distance_penalty = np.sqrt(curve_distance)
            reward -= curve_distance_penalty
            reward += 0.9 - self.y
        if self.steps % 200 == 0:
            print("Reward at step", self.steps, ":", reward)
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

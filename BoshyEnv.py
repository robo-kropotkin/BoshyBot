import subprocess
import keyboard
import numpy as np
import pygetwindow as gw
from os import path, getcwd
from time import sleep
from ReadWriteMemory import ReadWriteMemory
from gymnasium import Env


class BoshyEnv(Env):
    @classmethod
    def input_dims(cls):
        return 2

    def __init__(self, level=0):
        self.rwm = ReadWriteMemory()
        self.process = None
        self.process_handle = None
        self.im_address = self.x_pointer = self.y_pointer = 0
        self.x = self.y = 0.0
        self.level_width = 4000
        self.level_height = 500
        self.history = np.zeros((40, 10))

    def reset(self, seed=None, options=None):
        keyboard.release("z")
        keyboard.release("left")
        keyboard.release("right")
        keyboard.release("x")
        keyboard.press("r")
        sleep(0.03)
        keyboard.release("r")
        countdown = 2
        for i in range(0, countdown):
            print(countdown - i)
            sleep(1)
        self.read_process()
        return self.get_state()

    def get_state(self):
        state = np.array((self.x, self.y))
        assert state.shape == (BoshyEnv.input_dims(),)
        return state

    def run(self):
        self.process_handle = subprocess.Popen(path.join(getcwd(), r".\IWBTB\I Wanna Be The Boshy.exe"),
                                               stdin=subprocess.PIPE)
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
        self.x = self.process.read(self.x_pointer) / self.level_width
        self.y = self.process.read(self.y_pointer) / self.level_height
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

    def step(self, action, verbose=False):
        old_x = self.x
        # Coordinates bug out sometimes
        for i in range(100):
            self.read_process()
            if np.abs(self.x + self.y) < self.level_width + self.level_height and self.y != 0 and self.y != 8:
                break

        done = (self.y == 0 or self.y == 8)
        self.history[int(self.x * 40)][int(self.y * 10)] += 1
        reward = self.x + (self.x - old_x) * 1000 + 10 / self.history[int(self.x * 40)][int(self.y * 10)]
        observation = np.array((round(self.x, 3), round(self.y, 3)), dtype=np.float32)
        assert observation.shape == (BoshyEnv.input_dims(),)
        return observation, reward, done, False, {}

    def close(self):
        self.process_handle.terminate()

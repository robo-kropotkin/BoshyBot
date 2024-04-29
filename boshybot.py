# Boshy Bot
from ReadWriteMemory import ReadWriteMemory
from os import getcwd, path

import gymnasium as gym
import subprocess
import pyautogui
import keyboard
import random
import time

class BoshyBot:
    def __init__(self):
        self.rwm = ReadWriteMemory()
        self.deaths = 0
        self.deathTicker = 0
        self.gameOpen = False
        self.isRunnning = False
        self.process = 0
        self.base_address = 0
        self.death_tracker_pointer = 0
        self.absolute_x_pointer = 0

    def run(self):
        p = subprocess.Popen(path.join(getcwd(), r"IWBTB\I Wanna Be The Boshy.exe"))
        try:
            self.start_new_save()
            self.isRunnning = True
            self.read_process()
        except NameError as e:
            print(e)
        p.terminate()

    def read_process(self):
        self.process = self.rwm.get_process_by_name("I Wanna Be The Boshy.exe")
        self.process.open()
        preferred_image_address = 0x00400000
        self.base_address = preferred_image_address + 0x000598D4
        dt_offsets = [0x15C, 0x24, 0x4, 0x3C, 0x0, 0x28, 0x268]
        absx_offsets = [0x8D0, 0x30, 0x4C]
        self.death_tracker_pointer = self.process.get_pointer(self.base_address, dt_offsets)
        self.absolute_x_pointer = self.process.get_pointer(self.base_address, absx_offsets)

    def print_3_deaths(self):
        while self.deaths < 3:
            time.sleep(1)
            dt = self.process.read(self.death_tracker_pointer)
            if self.deathTicker != dt:
                self.deaths += 1
                print(self.deaths)
                self.deathTicker = dt

    def start_new_save(self):
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
        time.sleep(0.02)
        self.deaths = 0
        self.deathTicker = 0


if __name__ == "__main__":
    bot = BoshyBot()
    bot.run()

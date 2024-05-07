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

MAX_X = 4000
MAX_Y = 500
X_GRID_SIZE = 80
Y_GRID_SIZE = 10
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

def play(agent, action_delay, epoch=0):
    agent.min_loss = tensor(torch.inf)
    max_x = 0
    time_to_goal = timedelta.max
    iteration = 1
    done = False

    observation = env.reset()
    start_time = datetime.now()
    while not done and (datetime.now() - start_time).total_seconds() < epoch_duration:
        action = agent.choose_action(observation, iteration * (epoch + 1), verbose=True)
        if iteration < 40:
            action = 1

        keyboard_input(action)

        sleep(action_delay)
        agent.learn()
        # start_learning = datetime.now()
        # while (datetime.now() - start_learning).total_seconds() < action_delay:
        #     agent.learn()
        observation_, reward, done, info, _ = env.step(action, verbose=False)
        max_x = np.max([observation_[0], max_x])
        agent.store_transition(observation, action, reward, observation_, done)
        observation = observation_

        if iteration % 240 == 0 and True:
            print("Duration: ", datetime.now() - start_time)
            agent.learn(verbose=True)
            print("Reward: ", reward)
            print("===============")

        if iteration % 10 == 0:
            env.read_process()

        iteration += 1

        if keyboard.is_pressed("q"):
            break
    return agent.min_loss, max_x, time_to_goal

if __name__ == "__main__":
    env = BoshyEnv(max_x=MAX_X, max_y=MAX_Y)
    env.run()
    try:
        graph = False
        if graph:
            setup_graph()
        epochs = 1
        epoch_duration = 36000
        losses, x, times_to_goal = [], [], []
        action_delay = 0.075
        batch_size = 64
        mem_size = 100000
        lr = 1e-5
        exploration_factor = 0.1
        ef_multiplier = 5
        observation = env.reset()
        agent = BoshyAgent(gamma=0.9, batch_size=batch_size, n_actions=5,
                           input_dims=len(observation), lr=lr, graph=graph, x_grid=X_GRID_SIZE, y_grid=Y_GRID_SIZE,
                           max_mem_size=mem_size, exploration_factor=exploration_factor)
        best_weights = agent.main_network.state_dict()
        for epoch in range(epochs):
            max_x = 0
            time_to_goal = timedelta.max
            try:
                print("Begin epoch:", epoch + 1)
                current_loss, max_x, time_to_goal = play(agent, action_delay, epoch)
                agent.learn(verbose=True)
                print("End epoch:", epoch + 1)

            except (ReadWriteMemoryError, KeyboardInterrupt):
                print("Interrupted. How wude.")
                break
            finally:
                current_loss = agent.min_loss.cpu().detach().numpy().item()
                losses.append(current_loss)
                if current_loss < np.min(losses):
                    best_weights = agent.main_network.state_dict()
                x.append(max_x)
                times_to_goal.append(time_to_goal)
                if keyboard.is_pressed("q"):
                    print("Ok, you can quit")
                    break
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

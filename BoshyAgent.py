import numpy as np
import torch
from torch import tensor

from boshybot import QNet


class BoshyAgent:
    def __init__(self, gamma, lr, input_dims, initial_batch_size, batch_size, n_actions, max_mem_size=100000,
                 graph=False, exploration_factor=0.3):
        self.gamma = gamma
        self.lr = lr
        self.exploration_factor = exploration_factor
        self.graph = graph
        self.mem_size = max_mem_size

        self.batch_size = batch_size
        self.initial_batch_size = initial_batch_size
        self.mem_cntr = 0
        self.min_loss = tensor(torch.inf)

        self.Q_eval = QNet(self.lr, n_actions=n_actions, input_dims=input_dims)
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.downsampled_state_memory = np.zeros((self.mem_size, 2))
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.downsampled_state_memory[index] = np.array((int(state[0] * X_GRID_SIZE),
                                                         int(state[1] * Y_GRID_SIZE)), dtype=np.int32)
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1
        if self.mem_cntr % self.mem_size == 0:
            print("Memory Reset")

    def choose_action(self, observation, iteration, verbose=False):
        if iteration < self.initial_batch_size:
            return random.randint(0, 15)
        state = tensor(observation, dtype=torch.float32).to(self.Q_eval.device)
        downsampled_state = np.array((int(observation[0] * X_GRID_SIZE),
                                      int(observation[1] * Y_GRID_SIZE)), dtype=int)
        actions = self.Q_eval.forward(state)
        clipped_cntr = np.max((self.mem_cntr + 1, self.mem_size))
        state_memory = (np.sum(self.downsampled_state_memory == downsampled_state, axis=1) == 2)[:clipped_cntr]
        action_mem = tensor(self.action_memory[state_memory], dtype=torch.float32).to(self.Q_eval.device)
        memory_hist = torch.histc(action_mem, bins=actions.size(0), min=-1e-8, max=actions.size(0) + 1e-8) + 1
        ucb = self.exploration_factor / memory_hist

        action = torch.argmax(actions + ucb).item()

        if verbose and iteration % 1000 == 0:
            print("State:", state, "Downsampled:", downsampled_state)
            if action == torch.argmax(actions).item():
                print("^^Exploitation^^")
            else:
                print("~~Exploration~~")
            print("=================")
        return action

    def play(self, action_delay, epoch=0):
        self.min_loss = tensor(torch.inf)
        max_x = 0
        time_to_goal = timedelta.max
        iteration = 1
        done = False

        observation = env.reset()
        start_time = datetime.now()
        while not done and (datetime.now() - start_time).total_seconds() < epoch_duration:
            action = self.choose_action(observation, iteration * (epoch + 1), verbose=True)
            if iteration < 60:
                action = 2

            keyboard_input(action)

            start_learning = datetime.now()
            while (datetime.now() - start_learning).total_seconds() < action_delay:
                self.learn()

            observation_, reward, done, info, _ = env.step(action, verbose=False, epoch=epoch, tuning=tuning)
            max_x = np.max([observation_[0], max_x])
            if observation_[1] < 0.7 and time_to_goal == timedelta.max and iteration > batch_size:
                time_to_goal = datetime.now() - start_time

            self.store_transition(observation, action, reward, observation_, done)

            if iteration % 240 == 0 and True:
                action_delay = middle_action_delay
                print("Duration: ", datetime.now() - start_time)
                agent.learn(verbose=True)
                print("Reward: ", reward)
                print("===============")

            if iteration % 10 == 0:
                env.read_process()

            observation = observation_
            iteration += 1

            if keyboard.is_pressed("q"):
                break
        return self.min_loss, max_x, time_to_goal

    def learn(self, verbose=False):
        if self.mem_cntr < self.initial_batch_size:
            return
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        if self.mem_cntr < self.batch_size:
            batch = np.random.choice(max_mem, self.initial_batch_size, replace=False)
            batch_index = np.arange(self.initial_batch_size, dtype=np.int32)
        else:
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
            print(list(q_eval[:1].detach().cpu().numpy()), list(q_target[:1].detach().cpu().numpy()))
        loss.backward()
        self.Q_eval.optimizer.step()

        if self.graph:
            graph_net(self.Q_eval, 64 ** 2 * self.Q_eval.input_dims)

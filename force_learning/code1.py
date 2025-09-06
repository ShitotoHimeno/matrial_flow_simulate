#強化学習の勉強コードです。
#環境はグリッドワールド、エージェントはQ学習を使用します。
import pandas as pd
import numpy as np
import random

# 3x3のグリッドワールド環境の定義
class grid_env:
    def __init__(self):
        self.grid_size = 3
        self.num_states = self.grid_size * self.grid_size
        self.num_actions = 4  # up, down, left, right
        self.start_state = 0  # initial state
        self.goal_state = 8
        self.trap_state = 7

        self._state_to_coords = lambda s: (s // self.grid_size, s % self.grid_size)
        self._coords_to_state = lambda r,c: r * self.grid_size + c
        self.status = self.start_state

    def reset(self):
        self.state = self.start_state
        return self.state
    
    def step(self, action):
        r,c = self._state_to_coords(self.state)

        move = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

        dr, dc = move[action]
        new_r, new_c = r + dr, c + dc

        if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
            self.state = self._coords_to_state(new_r, new_c)
        
        else:
            next_state = self.state  # invalid move, stay in place

        if self.state == self.goal_state:
            reward = 10.0
            done = True
        
        elif self.state == self.trap_state:
            reward = -10.0
            done = True
        
        else:
            reward = -1.0
            done = False

        self.status = self.state
        return self.state, reward, done
    
#ランダムな行動でのデモンストレーション
print("ランダムな行動でのデモンストレーション")
env = grid_env()
status = env.reset()
done = False
while not done:
    action = np.random.choice(4)
    next_status, reward, done = env.step(action)
    print(f"Status: {status}, Action: {action}, Next Status: {next_status}, Reward: {reward}, Done: {done}")
    satus = next_status

# Q学習エージェントの定義
class qlearning_agent:
    def __init__ (self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((num_states, num_actions))

    def select_action(self,state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.Q[state])
        
    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

# Q学習エージェントの訓練
env = grid_env()
agent = qlearning_agent(env.num_states, env.num_actions)
num_episodes = 100
for episode in range (num_episodes):
    status = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(status)
        next_status, reward, done = env.step(action)
        agent.update(status, action, reward, next_status)
        status = next_status
        total_reward += reward

    if (episode + 1) % 50 == 0:
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# 学習結果の表示
print("\n学習結果の表示")
print("Learned Q-values:")
for s in range(env.num_states):
    print(f"State {s}: {agent.Q[s]}")

print("\n最適行動方針:")
for s in range(env.num_states):
    best_action = np.argmax(agent.Q[s])
    action_name = ["Up", "Down", "Left", "Right"][best_action]
    print(f"State {s}: {action_name}")
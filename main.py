from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import *
import itertools
from functools import lru_cache
from typing import Tuple
from time import time
from abc import abstractmethod, ABC
from random import choice, random
import torch
import numpy as np

env = make("hungry_geese", debug=True)
config = {"columns": 11, "rows": 7, "hunger_rate": 40, "min_food": 2}
n_actions = 4
n_dirs = 3

classifier = torch.nn.Sequential(
    torch.nn.Linear(3, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 3)
)
optim = torch.optim.Adam(classifier.parameters(), lr=1e-3)
loss = torch.nn.MSELoss()


def calc_td_loss(states, actions, rewards, gamma=0.99, check_shapes=False):
    next_states = states[1:]
    states = states[:-1]
    actions = actions[:-1]
    rewards = rewards[1:-1]
    if len(states) == 0:
        return

    states = torch.tensor(states, dtype=torch.float32)  # shape: [batch_size, state_size]
    actions = torch.tensor(actions, dtype=torch.long)  # shape: [batch_size]
    rewards = torch.tensor(rewards, dtype=torch.float32)  # shape: [batch_size]
    next_states = torch.tensor(next_states, dtype=torch.float32)  # shape: [batch_size, state_size]


    predicted_qvalues = classifier(states)
    # print(states.shape, actions.shape, rewards.shape)
    predicted_qvalues_for_actions = predicted_qvalues[range(states.shape[0]), actions]

    with torch.no_grad():
        predicted_next_qvalues = classifier(next_states)

    next_state_values = torch.max(predicted_next_qvalues, 1).values

    # print(next_state_values, rewards)

    target_qvalues_for_actions = rewards + next_state_values * gamma

    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)
    # добавляем регуляризацию на значения Q
    loss += 0.1 * predicted_qvalues_for_actions.mean()
    print(loss)
    return loss


def select_elites(states_batch, actions_batch, rewards_batch, total_rewards):
    states_batch = list(states_batch)
    actions_batch = list(actions_batch)
    rewards_batch = list(rewards_batch)
    total_rewards = list(total_rewards)

    max_rew = np.argmax(total_rewards, 1)

    elite_state = []
    elite_actions = []
    elite_rewards = []

    for i, mr in enumerate(max_rew):
        elite_state.append(states_batch[i][mr])
        elite_actions.append(actions_batch[i][mr])
        elite_rewards.append(rewards_batch[i][mr])

    # print("Elite")
    # print(elite_rewards)
    # print(elite_actions)

    return elite_state, elite_actions, elite_rewards


def calc_reward(session, count_geese):
    total_rewards = [0] * count_geese
    rewards_batch = []
    len_geese = [1] * count_geese
    start_dist = [-1] * count_geese

    reward_buffer = []
    for k in range(count_geese):
        for j in range(len(session)):
            current_length = len(session[j][0].observation.geese[k])

            if current_length == 0:
                total_rewards[k] -= 1.
                reward_buffer.append(total_rewards[k])
                break

            if start_dist[k] == -1:
                start_dist[k] = min_distance(session[j][0].observation.geese[k][0], session[j][0].observation.food, 11)
            else:
                new_dist = min_distance(session[j][0].observation.geese[k][0], session[j][0].observation.food, 11)
                total_rewards[k] += (start_dist[k] - new_dist) / 10.
                start_dist[k] = new_dist

            total_rewards[k] -= 0.05

            if current_length > len_geese[k]:
                total_rewards[k] += 1
                len_geese[k] = current_length

            elif current_length < len_geese[k]:
                total_rewards[k] -= 0.5
                len_geese[k] = current_length

            reward_buffer.append(total_rewards[k])
        rewards_batch.append(reward_buffer)
        reward_buffer = []
    return rewards_batch, total_rewards


class PossibleAction:
    NORTH = "NORTH"
    EAST = "EAST"
    SOUTH = "SOUTH"
    WEST = "WEST"

    def __init__(self):
        self.base_actions = [
            self.WEST,
            self.SOUTH,
            self.EAST,
            self.NORTH
        ]
        self.actions_ = [self.NORTH, self.EAST, self.WEST]  # F L R

    def possible_actions(self, last_action=None):
        if last_action is None:
            for i in range(n_dirs):
                self.actions_[i] = choice(self.base_actions)
                return

        self.actions_[0] = last_action

        forward_ind = self.base_actions.index(self.actions_[0])
        self.actions_[1] = self.base_actions[forward_ind - 1]
        self.actions_[2] = self.base_actions[(forward_ind + 1) if forward_ind < n_dirs else 0]


class AAgent(ABC):
    def __init__(self):
        self.actions = PossibleAction()
        self.last_act = None

        self.observation = None
        self.configuration = None

        self.player_index = None
        self.player_goose = None
        self.player_head = None

        self.all_geese = []
        self.food = []

        self.states_batch = []
        self.actions_batch = []
        self.epsilon = 0.5

    def get_base(self, obs_dict, config_dict):
        self.observation = Observation(obs_dict)
        self.configuration = Configuration(config_dict)

        self.player_index = self.observation.index
        self.player_goose = self.observation.geese[self.player_index]
        self.player_head = self.player_goose[0]

        self.all_geese = self.observation.geese
        self.food = self.observation.food

    def create_map(self):
        player_map = np.zeros((self.configuration.rows, self.configuration.columns))

        player_index = self.observation.index
        player_goose = self.observation.geese[player_index]
        player_row, player_column = row_col(player_goose[0], self.configuration.columns)
        player_map[player_row, player_column] = -1

        for pos in player_goose[1:]:
            r, c = row_col(pos, self.configuration.columns)
            player_map[r, c] = 1

        other_map = np.zeros((self.configuration.rows, self.configuration.columns))
        for i, goose_points in enumerate(self.observation.geese):
            if player_index == i:
                continue
            for pos in goose_points:
                r, c = row_col(pos, self.configuration.columns)
                other_map[r, c] = 1

        food_map = np.zeros((self.configuration.rows, self.configuration.columns))
        for pos in self.observation.food:
            r, c = row_col(pos, self.configuration.columns)
            food_map[r, c] = 1

    @abstractmethod
    def agent(self, obs_dict, config_dict):
        pass


class BaseAgent(AAgent):
    def agent(self, obs_dict, config_dict):
        self.get_base(obs_dict, config_dict)

        self.actions.possible_actions(last_action=self.last_act)

        act = choice([x for x in self.actions.actions_])
        self.last_act = act

        self.states_batch.append(np.array([self.player_head, *self.food]) / 77)
        self.actions_batch.append(self.actions.actions_.index(act))
        return act


class MLPAgent(AAgent):
    def agent(self, obs_dict, config_dict):
        self.get_base(obs_dict, config_dict)
        self.create_map()

        state = np.array([self.player_head, *self.food]) / 77
        self.states_batch.append(state)

        self.actions.possible_actions(self.last_act)

        q_value = classifier(torch.tensor([self.player_head, *self.food]) / 77).detach().numpy().tolist()

        act = self.actions.actions_[np.argmax(q_value)] if random() > self.epsilon else choice(self.actions.actions_)
        self.actions_batch.append(self.actions.actions_.index(act))
        self.last_act = act

        return act


def generate_session(env, agents: AAgent):
    env.reset()
    state = env.run([x.agent for x in agents])

    states = [x.states_batch for x in agents]
    actions = [x.actions_batch for x in agents]

    rewards, total_rewards = calc_reward(state, len(agents))
    return states, actions, rewards, total_rewards


base_agent = BaseAgent()

n_sessions = 5

log = []

for i in range(5):
    sessions = [generate_session(env, [MLPAgent(), MLPAgent(), MLPAgent()]) for _ in
                range(n_sessions)]

    states_batch, actions_batch, rewards_batch, total_rewards = zip(*sessions)

    elite_states, elite_actions, elite_rewards = select_elites(states_batch, actions_batch, rewards_batch,
                                                               total_rewards)

    for n in range(len(elite_states)):
        optim.zero_grad()
        loss = calc_td_loss(elite_states[n], elite_actions[n], elite_rewards[n])
        if loss is None:
            continue
        loss.backward()
        optim.step()

    # show_progress(rewards_batch, log, percentile, reward_range=[0, np.max(rewards_batch)])

    # if np.mean(rewards_batch) > 190:
    #     print("Принято!")
    #     break

env.render(mode="ipython", width=800, height=600)

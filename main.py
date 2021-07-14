from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
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


def calc_td_loss(states, actions, rewards, next_states, gamma=0.99, check_shapes=False):
    states = torch.tensor(states, dtype=torch.float32)  # shape: [batch_size, state_size]
    actions = torch.tensor(actions, dtype=torch.long)  # shape: [batch_size]
    rewards = torch.tensor(rewards, dtype=torch.float32)  # shape: [batch_size]
    next_states = torch.tensor(next_states, dtype=torch.float32)  # shape: [batch_size, state_size]

    predicted_qvalues = classifier(states)
    predicted_qvalues_for_actions = predicted_qvalues[range(states.shape[0]), actions]

    with torch.no_grad():
        predicted_next_qvalues = classifier(next_states)

    next_state_values = torch.max(predicted_next_qvalues, 1).values

    target_qvalues_for_actions = rewards + next_state_values * gamma

    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)
    # добавляем регуляризацию на значения Q
    loss += 0.1 * predicted_qvalues_for_actions.mean()

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
                start_dist[k] = food_distance(session[j][0].observation.geese[k][0], session[j][0].observation.food)
            else:
                new_dist = food_distance(session[j][0].observation.geese[k][0], session[j][0].observation.food)
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


def base_point(rc):
    return rc[0] * config["columns"] + rc[1]


def sum_by_elements(a, b):
    return tuple([a[i] + b[i] for i in range(2)])


def food_distance(position: int, food: [int]):
    columns = config["columns"]
    row, column = row_col(position, columns)
    return min(
        abs(row - food_row) + abs(column - food_column)
        for food_position in food
        for food_row, food_column in [row_col(food_position, columns)]
    )


class Actions:
    def __init__(self):
        self.base_actions = [
            "WEST",
            "NORTH",
            "EAST",
            "SOUTH"
        ]
        self.actions_ = {
            0: "EAST",  # F
            1: "SOUTH",  # L
            2: "WEST"  # R
        }

    @staticmethod
    def act2tuple(act: str):
        return {
            "EAST": (1, 0),
            "WEST": (-1, 0),
            "NORTH": (0, -1),
            "SOUTH": (0, 1)
        }[act]

    @staticmethod
    def tuple2act(t: Tuple[int, int]):
        if t[0] == 0:
            if t[1] == 1:
                return "NORTH"
            if t[1] == -1:
                return "SOUTH"
        if t[1] == 0:
            if t[0] == 1:
                return "EAST"
            if t[0] == -1:
                return "WEST"
        raise ValueError(f"Can't convert tuple {t}")

    @lru_cache(5)
    def possible_actions(self, last_action=None):
        if last_action is None:
            for i in range(n_dirs):
                self.actions_[i] = choice(self.base_actions)
                return

        self.actions_[0] = self.tuple2act(tuple(map((-1).__mul__, self.act2tuple(last_action))))

        forward_ind = self.base_actions.index(self.actions_[0])
        self.actions_[1] = self.base_actions[forward_ind - 1]
        self.actions_[2] = self.base_actions[(forward_ind + 1) if forward_ind < n_dirs else 0]


class AAgent(ABC):
    def __init__(self):
        self.actions = Actions()
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

    @abstractmethod
    def agent(self, obs_dict, config_dict): pass


class BaseAgent(AAgent):
    def agent(self, obs_dict, config_dict):
        self.get_base(obs_dict, config_dict)

        self.actions.possible_actions(last_action=self.last_act)

        act = choice([x for x in self.actions.actions_.values()])
        self.last_act = act

        self.states_batch.append(np.array([self.player_head, *self.food]) / 77)
        self.actions_batch.append(act)
        return act


class MLPAgent(AAgent):
    def agent(self, obs_dict, config_dict):
        self.get_base(obs_dict, config_dict)

        state = np.array([self.player_head, *self.food]) / 77
        self.states_batch.append(state)

        self.actions.possible_actions(self.last_act)

        q_value = classifier(torch.tensor([self.player_head, *self.food]) / 77).detach().numpy().tolist()

        act = self.actions.actions_[np.argmax(q_value)] if random() > self.epsilon else choice(self.actions.actions_)
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

n_sessions = 2

log = []

for i in range(1):
    sessions = [generate_session(env, [base_agent, MLPAgent(), MLPAgent()]) for _ in
                range(n_sessions)]

    states_batch, actions_batch, rewards_batch, total_rewards = zip(*sessions)

    print(states_batch)
    elite_states, elite_actions, elite_reward = select_elites(states_batch, actions_batch, rewards_batch,
                                                              total_rewards)

    s = []
    next_s = []
    a = []
    r = []

    for _ in range(len(elite_states)):
        s.append([x for x in elite_states[:len(elite_states)]])
        a.append([x for x in elite_actions[:len(elite_actions)]])
        next_s.append([x for x in elite_states[1:]])
        r.append([x for x in elite_reward[1:]])

    # optim.zero_grad()
    # calc_td_loss(s[0], [a], [r], [next_s]).backward()
    # optim.step()

    # show_progress(rewards_batch, log, percentile, reward_range=[0, np.max(rewards_batch)])

    # if np.mean(rewards_batch) > 190:
    #     print("Принято!")
    #     break

env.render(mode="ipython", width=800, height=600)

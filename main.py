from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import itertools
from functools import lru_cache
from typing import Tuple
from time import time
from abc import abstractmethod, ABC
from random import choice
import torch
import numpy as np

env = make("hungry_geese", debug=True)
config = {"columns": 11, "rows": 7, "hunger_rate": 40, "min_food": 2}
n_actions = 4

classifier = torch.nn.Sequential(
    torch.nn.Linear(3, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, n_actions)
)


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
        self.actions = [
            "EAST",
            "WEST",
            "NORTH",
            "SOUTH"
        ]

    @staticmethod
    def act2tuple(act: str):
        return {
            "EAST": (1, 0),
            "WEST": (-1, 0),
            "NORTH": (0, -1),
            "SOUTH": (0, 1)
        }[act]

    @lru_cache(5)
    def possible_actions(self, last_action=None):
        if last_action is None:
            return self.actions.copy()
        possible = self.actions.copy()
        possible.remove(last_action)
        return possible

    def actions_by_board(self, all_geese, head, last_action):
        actions = self.possible_actions(last_action)
        all_obstacles = list(itertools.chain(*all_geese))
        for act in actions:
            if row_col(head, 11) + self.act2tuple(act) in all_obstacles:
                actions.remove(act)
        return actions


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

        act = choice(list(self.actions.possible_actions(last_action=self.last_act)))
        self.last_act = act
        return act


class MLPAgent(AAgent):
    def agent(self, obs_dict, config_dict):
        self.get_base(obs_dict, config_dict)

        # player_row, player_column = row_col(self.player_head, self.configuration.columns)

        # food_row, food_column = row_col(self.food, self.configuration.columns)

        act = choice(list(self.actions.actions_by_board(self.all_geese, self.player_head, self.last_act)))
        if len(act) == 0:
            return choice(self.actions.actions)

        q_value = classifier(torch.tensor([self.player_head, *self.food]) / 77).detach().numpy().tolist()

        for q in sorted(q_value, reverse=True):
            if self.actions.actions[q_value.index(q)] in act:
                self.last_act = act
                return self.actions.actions[q_value.index(q)]

        raise Exception("None act")


def generate_session(env, agents):
    states, actions = [], []
    total_reward = 0

    env.reset()
    state = env.run(agents)

    print(calc_reward(state, len(agents)))


def calc_reward(session, count_geese):
    rewards = [0] * count_geese
    len_geese = [1] * count_geese

    for k in range(count_geese):
        for j in range(len(session)):
            current_length = len(session[j][0].observation.geese[k])

            if current_length == 0:
                rewards[k] -= 100

            rewards[k] -= 10
            if current_length > len_geese[k]:
                rewards[k] += 100
                len_geese[k] = current_length

            elif current_length < len_geese[k]:
                rewards[k] -= 50
                len_geese[k] = current_length
    print(rewards)


base_agent = BaseAgent()
# generate_session(env, [base_agent.agent, MLPAgent().agent, MLPAgent().agent])
n_sessions = 1
percentile = 50
log = []

for i in range(1):
    sessions = [generate_session(env, [base_agent.agent, MLPAgent().agent, MLPAgent().agent]) for _ in
                range(n_sessions)]

    # states_batch, actions_batch, rewards_batch = map(np.array, zip(*sessions))

    # elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)

    # classifier.fit(elite_states, elite_actions)

    # show_progress(rewards_batch, log, percentile, reward_range=[0, np.max(rewards_batch)])

    # if np.mean(rewards_batch) > 190:
    #     print("Принято!")
    #     break

env.render(mode="ipython", width=800, height=600)

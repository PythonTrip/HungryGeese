import gym
from gym import spaces
import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import *
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

N_DISCRETE_ACTIONS = 3
N_CHANNELS = 3
HEIGHT, WIDTH = 11, 7

CONFIG = {"columns": 11, "rows": 7, "hunger_rate": 40, "min_food": 2}

kaggle_env = make("hungry_geese", configuration=CONFIG, debug=False)

total_rewards = 0
len_geese = 1
start_dist = -1

epoch = 0


def calc_reward(goose, food, step):
    global total_rewards, len_geese, start_dist
    goose_len = len(goose)

    if goose_len == 0:
        total_rewards -= 2. if step > 15 else 15
        return total_rewards

    goose_head = goose[0]

    if start_dist == -1:
        start_dist = min_distance(goose_head, food, 11)
    else:
        new_dist = min_distance(goose_head, food, 11)
        total_rewards += (start_dist - new_dist) / start_dist * 2
        # start_dist = new_dist

    total_rewards -= 0.5

    if goose_len > len_geese:
        total_rewards += 15
        start_dist = min_distance(goose_head, food, 11)
        len_geese = goose_len

    elif goose_len < len_geese:
        total_rewards -= 0.5
        len_geese = goose_len
    return total_rewards


def create_map(observation: Observation, index):
    player_map = np.zeros((CONFIG["rows"], CONFIG["columns"]))

    player_index = index
    player_goose = observation.geese[player_index]
    if len(player_goose) == 0:
        return [player_map, player_map, player_map]

    player_row, player_column = row_col(player_goose[0], CONFIG["columns"])
    player_map[player_row, player_column] = 1

    for pos in player_goose[1:]:
        r, c = row_col(pos, CONFIG["columns"])
        player_map[r, c] = -1

    other_map = np.zeros((CONFIG["rows"], CONFIG["columns"]))
    for i, goose_points in enumerate(observation.geese):
        if player_index == i:
            continue
        for pos in goose_points:
            r, c = row_col(pos, CONFIG["columns"])
            other_map[r, c] = -1

    food_map = np.zeros((CONFIG["rows"], CONFIG["columns"]))
    for pos in observation.food:
        r, c = row_col(pos, CONFIG["columns"])
        food_map[r, c] = 1
    return [player_map, other_map, food_map]


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
            for i in range(N_DISCRETE_ACTIONS):
                self.actions_[i] = choice(self.base_actions)
                return

        self.actions_[0] = last_action

        forward_ind = self.base_actions.index(self.actions_[0])
        self.actions_[1] = self.base_actions[forward_ind - 1]
        self.actions_[2] = self.base_actions[(forward_ind + 1) if forward_ind < N_DISCRETE_ACTIONS else 0]


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, agents):
        super(CustomEnv, self).__init__()
        self.agents = agents
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(N_CHANNELS, WIDTH, HEIGHT), dtype=np.int)

        self.act = PossibleAction()
        self.last_act = None

    def step(self, action):
        self.act.possible_actions(self.last_act)
        actions = [self.act.actions_[action] for _ in range(len(self.agents))]
        observation = kaggle_env.step(actions)[0]

        obs = observation["observation"]

        state = np.array(create_map(obs, 0))

        reward = calc_reward(obs.geese[0], obs.food, obs.step)
        done = observation["status"]
        info = {}
        return state, reward, done, info

    def reset(self):
        global total_rewards, epoch

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, reward: {total_rewards}")

        epoch += 1
        total_rewards = 0
        observation: Observation = kaggle_env.reset(len(self.agents))[0].observation

        state = create_map(observation, 0)

        return np.array(state)  # reward, done, info can't be included

    def render(self, mode='ipython'):
        kaggle_env.render(mode="ipython", width=800, height=600)

    def close(self):
        pass


class Agent:
    def __init__(self, model):
        self.last_act = None
        self.act = PossibleAction()
        self.model = model

    def agent(self, obs_dict, config_dict):
        map = create_map(obs_dict, obs_dict.index)
        self.act.possible_actions(self.last_act)
        action, _ = self.model.predict(map, deterministic=True)
        act = self.act.actions_[action]
        self.last_act = act

        return act


env = CustomEnv([1] * 3)
# check_env(env)

state = env.reset()

agent = PPO("MlpPolicy", env)
agent.learn(25000)

kaggle_demo = make("hungry_geese", configuration=CONFIG, debug=True)
kaggle_demo.run([Agent(agent).agent, Agent(agent).agent])
kaggle_demo.render(mode="ipython", width=800, height=600)

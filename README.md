# Project create with support Russia association of AI

This project contain solution and ideas for solutions for Hungry Geese competition on Kaggle (https://www.kaggle.com/c/hungry-geese/overview/rules-of-play)

## Episode Objective
Survive the longest number of turns by eating food to stay alive, and by not running into other segments of your own goose or other agent's geese.

## Classifier

classifier = torch.nn.Sequential(...) - A model for learning agents to do true actions

optim = torch.optim.Adam(classifier.parameters(), lr=1e-3) - Optimizer

loss = torch.nn.MSELoss() - Loss function

## Select elite states with actions

Method selecting the best solution for the maximum reward in episodes
```python
def select_elites(states_batch, actions_batch, rewards_batch):
    max_rew = np.argmax(rewards_batch, 1)

    elite_state = []
    elite_actions = []
    for i, mr in enumerate(max_rew):
        elite_state.append(states_batch[i][mr])
        elite_actions.append(actions_batch[i][mr])
    return elite_state, elite_actions
```

## Claculation reward

Method calc reward by rules:
1) -0.1 points for step
2) -0.5 points for lose length
3) +1 points for add length
4) -1 points for die

```python
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
    return rewards
```

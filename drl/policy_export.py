import numpy as np

from stable_baselines3 import PPO

from env.retail_env import RetailShelfEnv

model = PPO.load(
    "results/ppo_retail"
)

env = RetailShelfEnv()

solutions = []

for _ in range(200):

    obs,_ = env.reset()

    action,_ = model.predict(obs)

    solutions.append(action)

solutions = np.array(solutions)

np.save(
    "results/policies.npy",
    solutions
)
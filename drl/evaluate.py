from stable_baselines3 import PPO

from env.retail_env import RetailShelfEnv

model = PPO.load(
    "results/ppo_retail"
)

env = RetailShelfEnv()

obs,_ = env.reset()

for _ in range(100):

    action,_ = model.predict(obs)

    obs,reward,done,trunc,info = env.step(action)

    print(info)

    if done:
        break
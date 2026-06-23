from env.retail_env import RetailShelfEnv

env = RetailShelfEnv()

obs, info = env.reset()

print(obs)
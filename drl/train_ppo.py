from stable_baselines3 import PPO
from env.retail_env import RetailShelfEnv

env = RetailShelfEnv()

obs, _ = env.reset()

print("Number of products:", env.num_products)
print("Observation shape:", obs.shape)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(
        net_arch=[256, 256]
    ),
    verbose=1
)

model.learn(total_timesteps=100000)

model.save("results/ppo_retail")
import os

save_path = os.path.join("results", "ppo_retail")

print("Saving model to:", os.path.abspath(save_path))

model.save(save_path)

print("Model saved successfully!")
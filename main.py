import subprocess

subprocess.run([
    "python",
    "-m",
    "drl.train_ppo"
])

subprocess.run([
    "python",
    "-m",
    "drl.policy_export"
])

subprocess.run([
    "python",
    "-m",
    "nsga2.optimize"
])
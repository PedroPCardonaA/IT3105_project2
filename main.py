import minatar
import numpy as np

env = minatar.Environment('breakout')
env.seed(42)  # Set seed separately
env.reset()

obs = env.state()
print("Initial Observation Shape:", obs.shape)

ep_return = 0
for step in range(100):
    action = np.random.randint(env.num_actions())
    reward, done = env.act(action)
    ep_return += reward
    if done:
        break

print("Episode Return after 100 steps or termination:", ep_return)
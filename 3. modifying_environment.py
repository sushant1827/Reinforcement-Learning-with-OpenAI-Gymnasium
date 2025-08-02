import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

# Start with a complex environment
env = gym.make("CarRacing-v3")
print(env.observation_space.shape)
# (96, 96, 3) - 96x96 RGB image

# Wrap it tot flatten the observation into a 1D array
wrapped_env = FlattenObservation(env)
print(wrapped_env.observation_space.shape)
# (27648,) - 96*96*3 = 27648 pixels flattened

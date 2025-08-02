import gymnasium as gym

# Discrete action spaces example
env = gym.make("CartPole-v1")

print(f"Action space: {env.action_space}") # Discrete(2) - left or right
print(f"Sample action: {env.action_space.sample()}") # Random action 0 or 1

# Box observation space (continuous values)
print(f"Observation space: {env.observation_space}") # Box space with 4 values
# Box Box([-4.8, -inf, -0.41887903, -inf], [4.8, inf, 0.41887903, inf], (4,), float32)

print(f"Sample observation: {env.observation_space.sample()}") # Random valid observation

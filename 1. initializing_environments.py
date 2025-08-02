import gymnasium as gym

# Create our training environment
# A cart with a pole on top that must be balanced
env = gym.make("CartPole-v1", render_mode="human")

# Reset the environment to its initial state
observation, info = env.reset(seed=42)
# observation: what the agent can "see" - cart position, velocity, pole angle, etc.
# info: extra debugging information (usually not needed for basic learning)

print(f"Initial observation: {observation}")
# Example output: 
# Initial observation: {'cart_position': 0.0, 'cart_velocity': 0.0, 'pole_angle': 0.0, 'pole_velocity': 0.0}

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action 0 = push cart left, 1 = push cart right
    action = env.action_space.sample()  # Random action for demonstration

    # Take the action in the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if we hit the time limit (500 steps)

    total_reward += reward
    episode_over = terminated or truncated

    print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Total Reward: {total_reward}")

# Close the environment when done
env.close()
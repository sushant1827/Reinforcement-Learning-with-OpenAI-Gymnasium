# -----------------------------------------------
# About the Environment: Blackjack
# -----------------------------------------------
# Clear rules: Get closer to 21 than the dealer without going over
# Simple observations: Your hand value, dealer’s showing card, usable ace
# Discrete actions: Hit (take card) or Stand (keep current hand)
# Immediate feedback: Win, lose, or draw after each hand

# This version uses infinite deck (cards drawn with replacement),
# so card counting won’t work - the agent must learn optimal basic strategy
# through trial and error.

# -----------------------------------------------
# Environment Details:
# -----------------------------------------------
# Observation: (player_sum, dealer_card, usable_ace)
# player_sum: Current hand value (4-21)
# dealer_card: Dealer’s face-up card (1-10)
# usable_ace: Whether player has usable ace (True/False)
# Actions: 0 = Stand, 1 = Hit
# Rewards: +1 for win, -1 for loss, 0 for draw
# Episode ends: When player stands or busts (goes over 21)

# -----------------------------------------------
# Building a Q-Learning Agent
# -----------------------------------------------
# Let’s build our agent step by step. We need functions for:

# Choosing actions (with exploration vs exploitation)
# Learning from experience (updating Q-values)
# Managing exploration (reducing randomness over time)

# -----------------------------------------------
# Exploration vs Exploitation
# -----------------------------------------------
# This is a fundamental challenge in RL:

# Exploration: Try new actions to learn about the environment
# Exploitation: Use current knowledge to get the best rewards

# We use epsilon-greedy strategy:

# With probability epsilon: choose a random action (explore)
# With probability 1-epsilon: choose the best known action (exploit)

# Starting with high epsilon (lots of exploration) and gradually reducing
# it (more exploitation as we learn) works well in practice.


from collections import defaultdict
import gymnasium as gym
import numpy as np


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """
        Initialize the Blackjack agent.

        Args:
            env             : The training environment
            learning_rate   : How quickly to update Q-values (0-1)
            initial_epsilon : Starting exploration rate (usually 1.0)
            epsilon_decay   : How much to reduce epsilon each episode
            final_epsilon   : Minimum exploration rate (usually 0.1)
            discount_factor : How much to value future rewards (0-1)
        """

        self.env = env

        # Q-table: Maps (state, action) pairs to rewards or Q-values
        # defaultdict automatically creates entries with default value 0.0 for new states/actions
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Choose an action based on epsilon-greedy strategy.

        Args:
            obs: Current observation (player_sum, dealer_card, usable_ace)

        Returns:
            Action index (0 for Stand, 1 for Hit)
        """

        # With probability epsilon: explore (random action)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        # With probability 1-epsilon: exploit (best known action)
        else:
            # Exploit: choose best known action
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """
        Update Q-values based on experience.

        Args:
            obs         : Current observation (player_sum, dealer_card, usable_ace)
            action      : Action index (0 for Stand, 1 for Hit)
            reward      : Reward for taking action
            terminated  : Whether the episode is over
            next_obs    : Next observation (player_sum, dealer_card, usable_ace)
        """

        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_values = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_values

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # This is the famous Bellman equation in action -
        # it says the value of a state-action pair should equal the
        # immediate reward plus the discounted value of the best next action.

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """
        Reduce exploration rate (epsilon) after each episode.
        """
        # Reduce epsilon by decay factor, but not below final_epsilon
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


# -----------------------------------------------
# Training the Agent
# -----------------------------------------------
# The process is:

# Reset environment to start a new episode
# Play one complete hand (episode), choosing actions and learning from each step
# Update exploration rate (reduce epsilon)
# Repeat for many episodes until the agent learns good strategy

# Training hyperparameters
learning_rate = 0.001  # How fast to learn (higher = faster but less stable)
n_episodes = 100_000  # Number of hands to practice
start_epsilon = 1.0  # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.05  # Always keep some exploration

# Create environment and agent
env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# Train the agent

from tqdm import tqdm  # Progress bar

for episode in tqdm(range(n_episodes)):
    # Reset environment to start a new episode
    obs, info = env.reset()
    done = False

    # Play one complete hand (episode)
    while not done:
        # Agent chooses an action (initially random, gradually more intelligent)
        action = agent.get_action(obs)

        # Take the action in the environment and observe the result
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Agent learns from the experience
        agent.update(obs, action, reward, terminated, next_obs)

        # Prepare for the next step
        obs = next_obs
        done = terminated or truncated

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()


# Analyzing Training Results

from matplotlib import pyplot as plt


def get_moving_avgs(arr, window, convolution_mode):
    """
    Compute moving average to smooth noisy data.

    Args:
        arr             : Input array of values
        window          : Size of the moving average window
        convolution_mode: Mode for np.convolve (e.g., 'valid', 'same')
    """
    return (
        np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode)
        / window
    )


# Smooth over a 500-episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(env.return_queue, rolling_length, "valid")
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(env.length_queue, rolling_length, "valid")
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error, rolling_length, "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.show()


# Testing our Trained Agent

# Test the trained agent
def test_agent(agent, env, num_episodes=1000):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

# Test your agent
test_agent(agent, env)
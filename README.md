# Reinforcement Learning with Gymnasium Examples

This repository contains basic examples demonstrating how to use the `gymnasium` library for setting up and interacting with Reinforcement Learning environments. It covers fundamental concepts such as environment initialization, understanding action and observation spaces, and modifying environments using wrappers.

## Project Structure

- `initializing_environments.py`: Demonstrates how to create, reset, and interact with a `gymnasium` environment (e.g., CartPole-v1) by taking random actions and observing the results.
- `action_observation_spaces.py`: Illustrates the different types of action and observation spaces available in `gymnasium` environments, including discrete and continuous (Box) spaces.
- `modifying_environment.py`: Shows how to use environment wrappers, specifically `FlattenObservation`, to modify the observation space of a complex environment (e.g., CarRacing-v3) for easier processing.

## Getting Started

To run these examples, you need to have Python and the `gymnasium` library installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sushant1827/Reinforcement-Learning-with-OpenAI-Gymnasium.git
    cd Reinforcement-Learning-with-OpenAI-Gymnasium
    ```

2.  **Install `gymnasium`:**
    ```bash
    pip install gymnasium
    pip install "gymnasium[box2d]" # For CarRacing-v3 environment
    ```

3.  **Run the examples:**
    ```bash
    python initializing_environments.py
    python action_observation_spaces.py
    python modifying_environment.py
    ```

## Contributing

Feel free to explore, modify, and contribute to these examples.

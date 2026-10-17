# Reinforcement Learning for Mobile Robot Navigation

A reinforcement learning project that trains a simulated differential-drive mobile robot to navigate toward a target using **Proximal Policy Optimization (PPO)**.

## Overview

This project explores how reinforcement learning can teach a mobile robot to reach a target through interaction with a physics-based simulation environment. The robot learns a navigation policy by receiving observations, selecting actions, and improving its behavior through reward feedback.

The project uses Python, PyBullet, Gymnasium, and Stable-Baselines3 to develop and evaluate the learning environment and control policy.

## Features

- **Physics-based simulation:** Simulates mobile robot movement using PyBullet.
- **Custom Gymnasium environment:** Defines observations, actions, rewards, and episode termination.
- **PPO reinforcement learning:** Trains a robot navigation policy using Stable-Baselines3.
- **Differential-drive control:** Controls the robot's movement through linear and angular motion.
- **Reward-based learning:** Encourages the robot to reduce its distance from the target.
- **Performance evaluation:** Tests the trained policy across multiple episodes.

## Technologies

- Python
- PyBullet
- Gymnasium
- Stable-Baselines3
- PPO (Proximal Policy Optimization)
- NumPy

## System Architecture

1. **Environment:** Simulates the robot and target position.
2. **Observation:** Provides the robot with navigation information, including distance and target direction.
3. **Policy:** PPO selects actions based on the current observation.
4. **Robot control:** Converts actions into movement commands.
5. **Reward function:** Evaluates progress toward the target.
6. **Training and evaluation:** Updates the policy and measures navigation performance.

## Training Workflow

```text
Initialize Simulation
        |
        v
Observe Robot and Target
        |
        v
PPO Selects an Action
        |
        v
Execute Robot Movement
        |
        v
Calculate Reward
        |
        v
Update Policy
        |
        v
Evaluate Navigation Performance
```

## Project Structure

```text
rl_robot/
├── environment/          # Custom robot environment, if separated
├── models/               # Saved trained models, if included
├── train.py              # Training script, if available
├── evaluate.py           # Evaluation script, if available
├── requirements.txt      # Python dependencies
└── README.md
```

*Adjust this structure to match the actual files in your repository.*

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rl_robot.git
cd rl_robot
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you do not have a dependency file yet:

```bash
pip install pybullet gymnasium stable-baselines3 numpy
```

### 4. Train the robot

Run the training script provided in the repository. For example:

```bash
python train.py
```

### 5. Evaluate the trained policy

Run the evaluation script if available:

```bash
python evaluate.py
```

Replace these example commands with the actual script names used in the project.

## Current Progress

The project has explored multiple versions of a differential-drive navigation environment, including changes to the observation space and reward-based learning.

One version using distance and target angle as observations trained with PPO for approximately **100,352 timesteps**, achieving an average episode reward of approximately **18.2** during the recorded training run.

These results are specific to that experiment and do not, by themselves, establish reliable navigation across all starting positions or environments.

## Future Improvements

- Improve success rates across randomized initial positions.
- Add obstacles and collision avoidance.
- Compare different observation spaces and reward functions.
- Evaluate generalization to unseen target locations.
- Visualize training rewards and navigation trajectories.
- Explore transfer from simulation to a physical mobile robot.

## Learning Objectives

- Understand reinforcement learning for robotic control.
- Implement custom Gymnasium environments.
- Train and evaluate PPO policies.
- Analyze reward functions and observation spaces.
- Develop a foundation for autonomous navigation and robotics research.

## Author

**PHON Napha**

Interested in artificial intelligence, reinforcement learning, robotics, and autonomous systems.

## License

Choose a license, such as the MIT License, before distributing the project publicly.

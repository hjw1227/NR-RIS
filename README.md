# SecureCoder: DRL-Based Robust Precoder Design Against Malicious NR-RIS

This repository contains the implementation of SecureCoder, an enhanced Proximal Policy Optimization (PPO) algorithm for robust precoder design in wireless networks against malicious Non-Reflective Intelligent Surface (NR-RIS) attacks. The solution leverages deep reinforcement learning to optimize communication precoders, ensuring resilience against intentional interference from malicious RIS elements.

## Table of Contents
- [Project Overview](#project-overview)
- [Code Structure](#code-structure)
- [Key Components](#key-components)
- [Dependencies](#dependencies)
- [Usage Guide](#usage-guide)
- [File Descriptions](#file-descriptions)

## Project Overview
The project addresses the challenge of designing resilient communication systems in the presence of malicious NR-RIS (Non-Reflective Intelligent Surface) attacks. Using deep reinforcement learning, the solution learns optimal precoding policies to maintain communication quality and security by:
- Modeling realistic wireless channels with Rice fading
- Implementing a priority experience replay buffer for efficient training
- Evaluating against multiple baselines (MRT, ZF, random beamforming)
- Calculating secrecy rate and outage probability metrics

## Code Structure
```
.
├── Env_test.py           # Adversarial RIS communication environment
├── norm.py               # Logarithmic normalization utility
├── normalization.py      # Dynamic mean-std normalization
├── ppo_continuous_cnn.py # PPO algorithm implementation
├── replaybuffer_con_cnn_per.py # Generate channel matrices
├── channel_generate.py   # Priority experience replay buffer
├── run.py                # Training script
└── run_test.py           # Evaluation script
```

## Key Components

### 1. Communication Environment (Env_test.py)
The `nd_ris` class simulates a RIS-assisted wireless environment with:
- **System Parameters**: 
  - `N` (RIS elements), `M` (base station antennas), `K` (users)
  - Path loss exponents, Rice factors for channel modeling
- **Core Methods**:
  - `reset()`: Initialize environment state
  - `get_state()`: Normalize channel state for agent input
  - `step()`: Execute action and return transitions
  - `step_test()`: Evaluate policy against attack scenarios
  - `generate_channel()`: Create Rice fading channels
  - `compute_reward()`: Calculate sum rate and secrecy rate
  - `ZF_precoding()`, `MRT_precoding()`: Baseline precoding strategies

### 2. Normalization Utilities
- **norm.py**: `LogNormalizer` for logarithmic transformation and normalization
- **normalization.py**: 
  - `RunningMeanStd`: Dynamically compute mean and standard deviation
  - `Normalization`: State normalization for stable training

### 3. PPO Algorithm (ppo_continuous_cnn.py)
- **Critic Network**: CNN-based value function approximation
- **Policy Optimization**: 
  - Beta distribution for continuous action space
  - Orthogonal initialization for stable training
  - Gradient clipping and learning rate decay
- **Key Methods**:
  - `choose_action()`: Sample actions from policy distribution
  - `update()`: PPO policy update with GAE
  - `evaluate()`: Determine deterministic action for evaluation

### 4. Priority Replay Buffer (replaybuffer_con_cnn_per.py)
- **Priority Sampling**: Store high-reward transitions for focused training
- **Mixed Sampling**: Balance between priority and random transitions
- **Key Functions**:
  - `store()`: Save transitions to main buffer
  - `update_priority_buffer()`: Update high-priority transitions
  - `numpy_to_tensor()`: Convert samples to tensor format

### 5. Training & Evaluation Scripts
- **run.py**: Main training loop with:
  - Channel data loading and initialization
  - Periodic policy evaluation and plotting
  - Model saving and result logging
- **run_test.py**: Evaluate trained policy against:
  - Malicious RIS attacks
  - Baseline precoding strategies
  - Secrecy rate and outage probability metrics

## Dependencies
- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- SciPy

## Usage Guide

### Training the Agent
```bash
python run.py --max_train_episode 2000000 --entropy_coef 0.005
```

### Evaluating the Policy
```bash
python run_test.py --times 5000
```

### Command Line Arguments
| Argument              | Default | Description                          |
|-----------------------|---------|--------------------------------------|
| `--max_train_episode` | 2000000 | Number of training episodes          |
| `--max_train_steps`   | 20      | Steps per training episode           |
| `--policy_dist`       | "Beta"  | Policy distribution (Beta/Gaussian)  |
| `--batch_size`        | 4000    | Training batch size                  |
| `--entropy_coef`      | 0.005   | Entropy regularization coefficient   |

## File Descriptions

### Env_test.py
Defines the communication environment with methods for:
- Channel generation using Rice fading models
- Malicious RIS attack simulation
- Secrecy rate calculation considering eavesdropper channels
- Evaluation against multiple attack scenarios and baselines

### ppo_continuous_cnn.py
Implements the PPO algorithm with:
- CNN-based actor-critic architecture
- Beta distribution for continuous action space
- Advanced training tricks (gradient clipping, orthogonal init, etc.)

### replaybuffer_con_cnn_per.py
Implements a priority experience replay buffer for:
- Storing high-priority transitions
- Mixed sampling to balance exploration-exploitation
- Efficient data loading for training

### channel_generate.py
Implements a wireless channel generation module for:
- Creating realistic communication channel matrices
- Generating channels for user, NR-RIS nodes
- Efficient data storage and loading for reproducible experiments
  
### norm.py & normalization.py
Provide normalization utilities for:
- Logarithmic transformation of channel data
- Dynamic mean and standard deviation calculation
- Stable state representation for RL training

This implementation enables robust policy learning in adversarial communication environments, providing a foundation for secure RIS-assisted wireless systems.

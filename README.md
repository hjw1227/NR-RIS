# SecureCoder: DRL-Based Robust Precoder Design Against Malicious NR-RIS

This repository contains the implementation of SecureCoder, an enhanced Proximal Policy Optimization (PPO) algorithm for robust precoder design in wireless networks against malicious NR-RIS attacks. The solution leverages deep reinforcement learning to optimize communication precoders, ensuring resilience against NR-RIS CRACK.

## Table of Contents
- [Project Overview](#project-overview)
- [Code Structure](#code-structure)
- [Key Components](#key-components)
- [Dependencies](#dependencies)
- [Usage Guide](#usage-guide)
- [File Descriptions](#file-descriptions)

## Project Overview
The project addresses the challenge of designing resilient communication systems in the presence of malicious NR-RIS attacks. Using deep reinforcement learning, the solution learns optimal precoding policies to maintain communication quality and security by:
- Modeling realistic wireless channels with Rician fading
- Implementing a priority experience replay buffer for efficient training
- Evaluating against multiple baselines (MRT, ZF, random beamforming)
- Calculating secrecy rate and secrecy outage probability(SoP) metrics

## Code Structure
```
.
├── Env_test.py           # Communication environment for test
├── Env.py                # Communication environment for training
├── norm.py               # Normalization utility
├── ppo_continuous_cnn.py # PPO algorithm implementation
├── replaybuffer_con_cnn_per.py # Priority experience replay buffer
├── channel_generate.py   #  Generate channel matrices
├── main.py                # Training script
└── main_test.py           # Evaluation script
```

## Key Components

### 1. Communication Environment (Env_test.py/Env.py)
The `NR_RIS_Env` class simulates a NR-RIS-assisted wireless environment with:
- **System Parameters**: 
  - `N` (RIS elements), `M` (base station antennas), `K` (users)
  - Path loss exponents, Rician factors for channel modeling
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
- **main.py**: Main training loop with:
  - Channel data loading and initialization
  - Periodic policy evaluation and plotting
  - Model saving and result logging
- **main_test.py**: Evaluate trained policy against:
  - Malicious RIS attacks
  - Baseline precoding strategies
  - Secrecy rate and SoP metrics

## Dependencies
- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- SciPy

## Usage Guide

### Training the Agent
```bash
python main.py --max_train_episode 300000 --entropy_coef 0.005
```

### Evaluating the Policy
```bash
python main_test.py --times 5000
Here is a saved model for M=32 N=64 --"actor_agent_32_64_4(entroy=0.005)(per_log2_ris2_no_fixed_rewardNorm).pth"
```

### Command Line Arguments
| Argument              | Default | Description                          |
|-----------------------|---------|--------------------------------------|
| `--max_train_episode` | 300000  | Number of training episodes          |
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
  
### norm.py
Provide normalization utilities for:
- Logarithmic transformation of channel data
- Dynamic mean and standard deviation calculation
- Stable state representation for RL training

This implementation enables robust policy learning in adversarial communication environments, providing a foundation for secure RIS-assisted wireless systems.

# SecureCoder: DRL-Based Robust Precoder Design Against Malicious NR-RIS

This repository contains the implementation of SecureCoder, an enhanced Proximal Policy Optimization (PPO) algorithm for robust precoder design in wireless networks against malicious Non-Reflective Intelligent Surface (NR-RIS) attacks. The solution leverages deep reinforcement learning to optimize communication precoders, ensuring resilience against intentional interference from malicious RIS elements.

## Table of Contents

- [Research Objective](#research-objective)
- [Methodology](#methodology)
- [SecureCoder Algorithm](#securecoder-algorithm)
- [Key Innovations](#key-innovations)
- [Code Structure](#code-structure)
- [Experimental Setup](#experimental-setup)
- [Evaluation Metrics](#evaluation-metrics)
- [How to Use](#how-to-use)

## Research Objective

The project addresses the challenge of designing robust precoders in wireless networks when facing malicious NR-RIS attacks. Malicious NR-RIS can intentionally distort communication channels, leading to performance degradation and security vulnerabilities. The goal is to develop a DRL-based approach that learns optimal precoding strategies to maintain communication quality and security under such adversarial conditions.

## Methodology

### Problem Formulation

We model the problem as a Markov Decision Process (MDP) where:
- **State**: Channel state information including direct links and RIS-assisted channels
- **Action**: Precoding matrix applied at the base station
- **Reward**: Secrecy rate considering both legitimate users and eavesdropper channels
- **Environment**: Wireless network with potential malicious NR-RIS interference

### Adversarial NR-RIS Model

The malicious NR-RIS is modeled to:
- Introduce intentional phase shifts and amplitude changes
- Disrupt channel reciprocity and coherence
- Mimic realistic attack patterns on communication links

## SecureCoder Algorithm

SecureCoder is an enhanced PPO algorithm with specific modifications to handle adversarial environments:

1. **Adversarial Robustness Modules**:
   - State normalization tailored for adversarial channel fluctuations
   - Reward shaping to prioritize resilience against NR-RIS attacks
   - Adaptive learning rate scheduling based on attack intensity

2. **Enhanced Experience Replay**:
   - Priority buffer emphasizing transitions under high attack intensity
   - Context-aware sampling to balance exploration and exploitation
   - Memory retention of adversarial scenarios for robust generalization

3. **Policy Optimization**:
   - Orthogonal initialization for stable policy convergence
   - Gradient clipping to prevent overfitting to specific attack patterns
   - Entropy regularization for policy diversity under uncertainty

## Key Innovations

### 1. Adversarial State Representation

The algorithm processes channel states through:
- Logarithmic normalization for skewed adversarial channel distributions
- Dynamic sliding window normalization to adapt to changing attack patterns
- Complex-valued state encoding to preserve phase information critical for precoding

### 2. Robust Reward Function

The reward function is designed to:
- Maximize legitimate user sum rate
- Minimize eavesdropper channel quality
- Penalize performance degradation under NR-RIS attacks
- Balance short-term reward with long-term security stability

### 3. Attack Simulation Framework

The environment includes:
- Realistic NR-RIS attack models (phase flipping, amplitude distortion)
- Configurable attack intensity and patterns
- Comparison against multiple attack scenarios (random, targeted, adaptive)

## Code Structure

```
.
├── Env_test.py           # Adversarial RIS communication environment
├── norm.py               # Adversarial-aware normalization utilities
├── replaybuffer_con_cnn.py # Priority replay for adversarial scenarios
├── ppo_continuous_cnn.py # SecureCoder PPO implementation
├── test_policy.py        # Evaluation against malicious RIS attacks
├── channel_data/         # Precomputed channel datasets with attack profiles
└── utils/                # Attack simulation and metrics calculation
```

### Core Components

- **Adversarial Environment (Env_test.py)**:
  - Models NR-RIS attacks with configurable intensity
  - Simulates channel distortion under malicious manipulation
  - Computes secrecy rate and outage probability metrics

- **Robust Replay Buffer (replaybuffer_con_cnn.py)**:
  - Prioritizes transitions from high-attack scenarios
  - Maintains a separate buffer for adversarial experience
  - Implements context-based sampling for robust training

- **SecureCoder PPO (ppo_continuous_cnn.py)**:
  - Adversarial-aware policy network architecture
  - Enhanced gradient processing for unstable environments
  - Attack-adaptive exploration-exploitation balance

## Experimental Setup

### Simulation Parameters

- **System Configuration**:
  - Base station antennas (M): 32
  - RIS elements (N): 64
  - Users (K): 4
  - Bandwidth (B): 10 MHz

- **Channel Model**:
  - Rice fading with configurable K factors
  - Path loss exponents for different links
  - Malicious NR-RIS attack models (phase/amplitude distortion)

- **Attack Scenarios**:
  - Random phase perturbation
  - Targeted channel nulling
  - Adaptive attack based on observed precoding patterns

## Evaluation Metrics

1. **Secrecy Rate (bps/Hz)**: Difference between legitimate sum rate and eavesdropper rate
2. **Outage Probability**: Probability of secrecy rate falling below a threshold
3. **Robustness Index**: Normalized performance under attack vs. clean channel
4. **Attack Resistance Time**: Time until performance degradation under sustained attack

## How to Use

### Training SecureCoder

```bash
python train.py --max_train_episode 2000000 --entropy_coef 0.035 --attack_intensity 0.6
```

### Evaluating Against Malicious RIS

```bash
python test_policy.py --attack_scenario targeted --evaluation_episodes 1000
```

### Command-line Arguments

| Argument               | Default | Description                                  |
|------------------------|---------|--------------------------------------------|
| `--attack_intensity`   | 0.5     | Intensity of malicious NR-RIS attack        |
| `--attack_scenario`    | random  | Attack type (random/targeted/adaptive)      |
| `--robustness_mode`    | full    | Training mode for robustness (full/lite)    |
| `--priority_ratio`     | 0.3     | Ratio of adversarial experience in replay  |

For detailed usage and parameter tuning, refer to the [documentation](docs/usage.md) or the inline comments in the code.

## Paper Reference

For more details on the algorithm and experimental results, please refer to our paper:

"SecureCoder: A DRL-Based Approach for Robust Precoder Design Against Malicious Non-Reflective Intelligent Surfaces"  
(To appear in IEEE Transactions on Information Forensics and Security, 2025)

```bibtex
@article{securecoder2025,
  title={SecureCoder: A DRL-Based Approach for Robust Precoder Design Against Malicious Non-Reflective Intelligent Surfaces},
  author={Your Name and Co-Authors},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

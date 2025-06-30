import torch
import numpy as np
import argparse

import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from utils.ppo_continuous_cnn import PPO_continuous_cnn
import Env_test

# Communication parameters
M = 32  # Number of base station antennas
N = 64  # Number of RIS elements
K = 4  # Number of users

# Initialize environments
env_evaluate = Env_test.nr_ris(N, M)
malicious = True

def test_policy(args, env, times):
    """Evaluate trained PPO policy against various baselines"""
    # Initialize metrics storage
    test_reward_random_ris_mrt = []
    test_reward_random_ris_zf = []
    test_reward_direct_mrt = []
    test_reward_mrt = []
    test_reward_direct_zf = []
    test_reward_zf = []
    test_reward_defend = []
    test_reward_random_bf = []
    test_reward_secrecy = []
    test_reward_eve_secure = []
    test_reward_eve_random_bf = []
    test_reward_eve_crack_mrt = []
    test_reward_eve_crack_zf = []

    # Load pre-generated channel data
    data = np.load('channel_data2_ris2_32_64_4.npz')
    args.state_dim = 2 * M * K
    args.action_dim = 2 * M * K
    print(f"State dimension: {args.state_dim}")
    print(f"Action dimension: {args.action_dim}")
    args.max_action = float(1)

    # Initialize and load trained agent
    agent = PPO_continuous_cnn(args)
    agent.load_model('agent_32_64_4(entroy=0.004)(per_log2_ris2_no_fixed_rewardNorm)')

    # Extract channel matrices
    Hur_all = data['Hur_all']
    Hub_all = data['Hub_all']
    Hrb_all = data['Hrb_all']

    # Initialize RIS configuration
    if malicious:
        env.get_NR_array(Hur_all, Hub_all, Hrb_all,True)
    else:
        env.get_NR_array(Hur_all, Hub_all, Hrb_all, False)


    # Evaluate policy over multiple episodes
    for ep in range(times):
        # Initialize episode rewards
        episode_reward_random_ris_mrt = 0
        episode_reward_random_ris_zf = 0
        episode_reward_direct_mrt = 0
        episode_reward_mrt = 0
        episode_reward_direct_zf = 0
        episode_reward_zf = 0
        episode_reward_defend = 0
        episode_reward_random_bf = 0
        episode_reward_secrecy = 0
        episode_reward_eve_secure = 0
        episode_reward_eve_random_bf = 0
        episode_reward_eve_crack_mrt = 0
        episode_reward_eve_crack_zf = 0

        # Run single episode
        for step in range(1):
            # Generate new channel realization
            env.get_NR_array(Hur_all, Hub_all, Hrb_all)
            state = env.reset()

            # Get action from trained policy
            action = agent.evaluate(state)

            # Step environment and collect rewards
            state_, reward_secrecy, reward_eve_secure, reward_eve_random_bf, reward_eve_crack_mrt, reward_eve_crack_zf, \
                reward_defend, reward_random_bf, reward_random_mrt, reward_random_zf, reward_mrt, reward_direct_mrt, reward_zf, reward_direct_zf = env.step_test(
                action, step)

            # Accumulate rewards
            episode_reward_random_ris_mrt += reward_random_mrt
            episode_reward_random_ris_zf += reward_random_zf
            episode_reward_mrt += reward_mrt
            episode_reward_direct_mrt += reward_direct_mrt
            episode_reward_zf += reward_zf
            episode_reward_direct_zf += reward_direct_zf
            episode_reward_defend += reward_defend
            episode_reward_random_bf += reward_random_bf
            episode_reward_eve_secure += reward_eve_secure
            episode_reward_eve_random_bf += reward_eve_random_bf
            episode_reward_eve_crack_mrt += reward_eve_crack_mrt
            episode_reward_eve_crack_zf += reward_eve_crack_zf
            episode_reward_secrecy += reward_secrecy

        # Store episode rewards
        test_reward_random_ris_mrt.append(episode_reward_random_ris_mrt)
        test_reward_random_ris_zf.append(episode_reward_random_ris_zf)
        test_reward_mrt.append(episode_reward_mrt)
        test_reward_direct_mrt.append(episode_reward_direct_mrt)
        test_reward_zf.append(episode_reward_zf)
        test_reward_direct_zf.append(episode_reward_direct_zf)
        test_reward_defend.append(episode_reward_defend)
        test_reward_random_bf.append(episode_reward_random_bf)
        test_reward_eve_secure.append(episode_reward_eve_secure)
        test_reward_eve_random_bf.append(episode_reward_eve_random_bf)
        test_reward_eve_crack_mrt.append(episode_reward_eve_crack_mrt)
        test_reward_eve_crack_zf.append(episode_reward_eve_crack_zf)
        test_reward_secrecy.append(episode_reward_secrecy)

    # Plot results
    fig, ax = plt.subplots(2, 1, figsize=(6, 9))
    plt.subplots_adjust(left=0.2, right=0.8, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)
    ax1, ax2 = ax.flatten()

    # Plot MRT comparison
    ax1.plot(test_reward_defend, color='blue')
    ax1.plot(test_reward_random_bf, color='grey')
    ax1.plot(test_reward_mrt, color='red')
    ax1.legend(['Secure_Code', 'Random_Code', 'CRACK_MRT_Rondom_RIS'])
    ax1.set_title('MRT Precoding')
    ax1.set_xlabel('Time slot')
    ax1.set_ylabel('Sum rate')

    # Plot ZF comparison
    ax2.plot(test_reward_defend, color='blue')
    ax2.plot(test_reward_zf, color='red')
    ax2.legend(['Secure_Code', 'CRACK_ZF_Random_RIS'])
    ax2.set_title('ZF Precoding')
    ax2.set_xlabel('Time slot')
    ax2.set_ylabel('Sum rate')

    # Print results
    print("Secure_Code:", np.sum(test_reward_defend) / times)
    print("Random_Code:", np.sum(test_reward_random_bf) / times)
    print("NO_CRACK_MRT:", np.sum(test_reward_direct_mrt) / times)
    print("CRACK_MRT:", np.sum(test_reward_mrt) / times)
    print("MRT_Random_RIS:", np.sum(test_reward_random_ris_mrt) / times)
    print("N0_CRACK_ZF:", np.sum(test_reward_direct_zf) / times)
    print("CRACK_ZF:", np.sum(test_reward_zf) / times)
    print("ZF_Random_RIS:", np.sum(test_reward_random_ris_zf) / times)
    print('eve_secure_rate:', np.sum(test_reward_eve_secure) / times)
    print('eve_random_bf_rate:', np.sum(test_reward_eve_random_bf) / times)
    print('eve_crack_mrt_rate:', np.sum(test_reward_eve_crack_mrt) / times)
    print('eve_crack_zf_rate:', np.sum(test_reward_eve_crack_zf) / times)
    print('secrecy_rate:', np.sum(test_reward_secrecy) / times)
    print('sop_secure:', np.float64(env.out_secure / (4 * times)))
    print('sop_random_bf:', np.float64(env.out_random_bf / (4 * times)))
    print('sop_crack_mrt:', np.float64(env.out_crack_mrt / (4 * times)))
    print('sop_crack_zf:', np.float64(env.out_crack_zf / (4 * times)))

    # Save and display plot
    plt.savefig('./test_32_64_malicious.pdf', format='pdf', bbox_inches='tight', dpi=600, pad_inches=0.0)
    plt.show()


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_episode", type=int, default=int(2000000),
                        help="Maximum number of training episodes")
    parser.add_argument("--max_train_steps", type=int, default=int(20),
                        help="Maximum number of training steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=200,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Model save frequency")
    parser.add_argument("--policy_dist", type=str, default="Beta", help="Policy distribution (Beta or Gaussian)")
    parser.add_argument("--batch_size", type=int, default=4000, help="Batch size for training")
    parser.add_argument("--mini_batch_size", type=int, default=2000, help="Minibatch size for training")
    parser.add_argument("--hidden_width", type=int, default=512, help="Neural network hidden layer size")
    parser.add_argument("--lr_a", type=float, default=1e-4, help="Actor learning rate")
    parser.add_argument("--lr_c", type=float, default=1e-4, help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.35, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="Number of policy optimization epochs")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Use advantage normalization")
    parser.add_argument("--entropy_coef", type=float, default=0.035, help="Entropy regularization coefficient")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Use learning rate decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Use gradient clipping")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Use orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Set Adam optimizer epsilon")
    parser.add_argument("--use_tanh", type=float, default=True, help="Use tanh activation function")
    parser.add_argument("--use_gru", type=bool, default=True, help="Use GRU in network")
    parser.add_argument("--hidden_dim", type=int, default=512, help="GRU hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Run policy evaluation
    test_policy(args, env_evaluate, 5000)

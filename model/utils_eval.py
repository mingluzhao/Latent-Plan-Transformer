import os
import numpy as np
import torch
import gym
from constants import REF_MAX_SCORE, REF_MIN_SCORE, D4RL_DATASET_STATS
import d4rl 

def get_d4rl_dataset_stats(env_d4rl_name):
    """
    Retrieve the precomputed mean and standard deviation for a given D4RL dataset.

    Args:
        env_d4rl_name (str): Name of the D4RL environment dataset.

    Returns:
        dict: A dictionary containing 'state_mean' and 'state_std'.
    """
    return D4RL_DATASET_STATS[env_d4rl_name]


def get_d4rl_normalized_score(score, env_name):
    """
    Calculate the normalized D4RL score for a given environment and raw score.

    Args:
        score (float): The raw score obtained from the environment.
        env_name (str): The name of the environment.

    Returns:
        float: The normalized score.
    """
    env_key = env_name.split('-')[0].lower()
    if 'kitchen' in env_name or "maze2d" in env_name:
        env_key = env_name

    if env_key not in REF_MAX_SCORE:
        raise ValueError(f"No reference score for {env_key} environment to calculate D4RL score.")

    return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])


def get_datasets_mean_sd(dataset):
    """
    Calculate the mean and standard deviation of observations in the dataset.

    Args:
        dataset (dict): The dataset containing observations.

    Returns:
        tuple: A tuple containing state mean and state standard deviation.
    """
    states = []
    for obs in dataset["observations"]:
        states.extend(obs)
    states = np.vstack(states)
    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0) + 1e-6
    return state_mean, state_std


def evaluate_env(model, device, env_name, eval_dataset, env_targets, scale, num_eval_ep, seed=None, omega_cg=1):
    """
    Evaluate the given model in the specified environment.

    Args:
        model: The trained model to evaluate.
        device (torch.device): The device to run the model on.
        env_name (str): Name of the environment.
        eval_dataset (str): The evaluation dataset.
        env_targets (float): The target return for the environment.
        scale (float): Scale factor for returns.
        num_eval_ep (int): Number of evaluation episodes.
        seed (int, optional): Random seed for reproducibility.
        omega_cg (float, optional): Coefficient for classifier guidance.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    # if torch.cuda.device_count() > 1:
    #     model = model.module
    model = model.to(device)
    model.eval()

    # Initialize the environment
    if 'hopper' in env_name:
        env = gym.make('Hopper-v3')
    elif 'halfcheetah' in env_name:
        env = gym.make('HalfCheetah-v3')
    elif 'walker2d' in env_name:
        env = gym.make('Walker2d-v3')
    elif "antmaze" in env_name:
        env = gym.make(f'{env_name}-v2')
    elif "maze2d" in env_name:
        env = gym.make("maze2d-medium-v1")
        env = gym.make(f'{env_name}-v1')
    elif "kitchen" in env_name:
        env = gym.make(f'{env_name}-v0')
    else:
        raise NotImplementedError(f"Environment '{env_name}' is not supported.")

    max_ep_len = env._max_episode_steps
    target_return = env_targets / scale

    # Get dataset name
    if "antmaze" in env_name:
        dataset_name = f'{env_name}-v2'
    elif "maze2d" in env_name:
        dataset_name = f'{env_name}-v1'
    elif 'kitchen' in env_name:
        dataset_name = f'{env_name}-v0'
    else:
        dataset_name = f'{env_name}-{eval_dataset}-v2'


    # Get dataset statistics
    env_data_stats = get_d4rl_dataset_stats(dataset_name)
    state_mean = np.array(env_data_stats['state_mean'])
    state_std = np.array(env_data_stats['state_std'])
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    state_mean = torch.from_numpy(state_mean).to(device=device, dtype=torch.float32)
    state_std = torch.from_numpy(state_std).to(device=device, dtype=torch.float32)

    # Set seeds for reproducibility
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # Evaluation loop
    episodes_return = []
    episodes_length = []
    batch_size = 1

    for episode in range(num_eval_ep):
        # Initialize episode
        episode_return = 0
        episode_length = 0
        state = env.reset()
        target_return_tensor = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)

        states = torch.from_numpy(state).reshape(batch_size, 1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((batch_size, 1, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        # Infer latent variable z
        z, _ = model.infer_z_given_y(y=target_return_tensor, omega_cg=omega_cg)

        for t in range(max_ep_len):
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            timesteps = torch.tensor([t], device=device, dtype=torch.long)

            _, action_preds, _ = model.generate(
                z=z,
                states=(states - state_mean) / state_std,
                actions=actions,
                timesteps=timesteps
            )

            action = action_preds[:, -1, :].squeeze(1)
            actions = torch.cat((actions, action.unsqueeze(1)), dim=1)
            action_np = action.detach().cpu().numpy().squeeze()

            next_state, reward, done, _ = env.step(action_np)

            cur_state = torch.from_numpy(next_state).reshape(batch_size, 1, state_dim).to(device=device, dtype=torch.float32)
            states = torch.cat([states, cur_state], dim=1)
            rewards[-1] = reward

            episode_return += reward
            episode_length += 1

            if done:
                break

        episodes_return.append(episode_return)
        episodes_length.append(episode_length)
        print(f"Episode {episode + 1}: Return = {episode_return}, Length = {episode_length}")

    avg_reward = np.mean(episodes_return)
    avg_ep_len = np.mean(episodes_length)
    norm_score = get_d4rl_normalized_score(avg_reward, dataset_name) * 100

    model.train()

    print(f"Average Reward: {avg_reward}, Average Episode Length: {avg_ep_len}")

    return {
        "eval/avg_reward": avg_reward,
        "eval/avg_ep_len": avg_ep_len,
        "eval/norm_score": norm_score
    }

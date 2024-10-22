"""
Data Processing Script for Datasets

This script processes various datasets from OpenAI Gym environments and saves them in a structured format.

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the CC BY-NC license found in the LICENSE.md file in the root directory of this source tree.
"""

import os
import gym
import numpy as np
import d4rl 
import collections
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset, DatasetDict


def split_into_trajectories(dones_float, observations, next_observations, actions, rewards):
    """
    Splits transitions into trajectories based on done signals.

    Args:
        dones_float (np.array): Array indicating episode terminations (1.0 if done, else 0.0).
        observations (np.array): Array of observations.
        next_observations (np.array): Array of next observations.
        actions (np.array): Array of actions.
        rewards (np.array): Array of rewards.

    Returns:
        List[Dict]: List of trajectories, where each trajectory is a dictionary containing observations,
                    actions, rewards, next observations, and terminals.
    """
    trajs = [defaultdict(list)]
    for i in tqdm(range(len(observations))):
        trajs[-1]['observations'].append(observations[i])
        trajs[-1]['actions'].append(actions[i])
        trajs[-1]['rewards'].append(rewards[i])
        trajs[-1]['next_observations'].append(next_observations[i])
        trajs[-1]['terminals'].append(dones_float[i])

        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append(defaultdict(list))

    for traj in trajs:
        for key in traj:
            traj[key] = np.array(traj[key])

    return trajs


def get_dataset_mean_std(dataset):
    """
    Computes the mean and standard deviation of states in the dataset.

    Args:
        dataset (Dataset): HuggingFace Dataset object containing trajectories.

    Returns:
        Tuple[np.array, np.array]: Mean and standard deviation of the states.
    """
    states = []
    for obs in dataset['observations']:
        states.extend(obs)
    states = np.vstack(states)
    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0) + 1e-6
    return state_mean, state_std


def process_mujoco_datasets():
    """
    Processes MuJoCo datasets and saves them to disk.
    """
    for env_name in ['halfcheetah', 'hopper', 'walker2d']:
        for dataset_type in ['medium', 'medium-replay']:
            name = f"{env_name}-{dataset_type}-v2"
            env = gym.make(name)
            dataset = env.get_dataset()
            N = dataset['rewards'].shape[0]
            data_ = collections.defaultdict(list)
            use_timeouts = 'timeouts' in dataset
            episode_step = 0
            paths = []

            for i in range(N):
                done_bool = bool(dataset['terminals'][i])
                final_timestep = dataset['timeouts'][i] if use_timeouts else (episode_step == env._max_episode_steps - 1)
                for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                    data_[k].append(dataset[k][i])
                if done_bool or final_timestep:
                    episode_step = 0
                    episode_data = {k: np.array(v) for k, v in data_.items()}
                    paths.append(episode_data)
                    data_ = collections.defaultdict(list)
                episode_step += 1

            returns = np.array([np.sum(p['rewards']) for p in paths])
            num_samples = np.sum([len(p['rewards']) for p in paths])
            print(f"Processing {name}:")
            print(f"Number of samples collected: {num_samples}")
            print(f"Number of trajectories: {len(paths)}")
            print(f"Trajectory returns: mean={np.mean(returns):.2f}, std={np.std(returns):.2f}, "
                  f"max={np.max(returns):.2f}, min={np.min(returns):.2f}")

            # Prepare dataset for saving
            dataset_list = [{
                'observations': traj['observations'],
                'actions': traj['actions'],
                'rewards': traj['rewards'],
                'dones': traj['terminals']
            } for traj in paths]

            dataset_dict = {key: [d[key] for d in dataset_list] for key in dataset_list[0]}
            dataset_hf = Dataset.from_dict(dataset_dict)
            dataset_hf = DatasetDict({'train': dataset_hf})
            print(dataset_hf)

            # Save dataset to disk
            directory_path = f'{name}'
            os.makedirs(directory_path, exist_ok=True)
            dataset_hf.save_to_disk(directory_path)


def process_antmaze_datasets():
    """
    Processes AntMaze datasets and saves them to disk.
    """
    env_names = [
        'antmaze-umaze-v2',
        'antmaze-medium-diverse-v2',
        'antmaze-large-diverse-v2',
    ]
    for env_name in env_names:
        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)
        env.close()

        dones_float = np.zeros_like(dataset['rewards'])
        for i in range(len(dones_float) - 1):
            if (np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6 or
                    dataset['terminals'][i] == 1.0):
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        dones_float[-1] = 1

        trajectories = split_into_trajectories(
            dones_float,
            dataset['observations'],
            dataset['next_observations'],
            dataset['actions'],
            dataset['rewards'],
        )

        returns = np.array([np.sum(traj['rewards']) for traj in trajectories])
        lengths = np.array([len(traj['rewards']) for traj in trajectories])
        num_samples = np.sum(lengths)
        print(f"Processing {env_name}:")
        print(f"Number of samples collected: {num_samples}")
        print(f"Number of trajectories: {len(trajectories)}")
        print(f"Trajectory returns: mean={np.mean(returns):.2f}, std={np.std(returns):.2f}, "
              f"max={np.max(returns):.2f}, min={np.min(returns):.2f}")
        print(f"Trajectory lengths: mean={np.mean(lengths):.2f}, std={np.std(lengths):.2f}, "
              f"max={np.max(lengths)}, min={np.min(lengths)}")

        # Prepare dataset for saving
        dataset_list = [{
            'observations': traj['observations'],
            'actions': traj['actions'],
            'rewards': traj['rewards'],
            'dones': traj['terminals']
        } for traj in trajectories]

        dataset_dict = {key: [d[key] for d in dataset_list] for key in dataset_list[0]}
        dataset_hf = Dataset.from_dict(dataset_dict)
        dataset_hf = DatasetDict({'train': dataset_hf})
        print(dataset_hf)

        # Save dataset to disk
        directory_path = f'dataset/{env_name}'
        os.makedirs(directory_path, exist_ok=True)
        dataset_hf.save_to_disk(directory_path)


def customize_qlearning_dataset(env):
    """
    Customizes the q-learning dataset for Maze2D environments.

    Args:
        env (gym.Env): The Gym environment.

    Returns:
        Dict: Customized dataset with observations, actions, next observations, rewards, and terminals.
    """
    dataset = env.get_dataset()
    N = dataset['rewards'].shape[0]
    obs_, next_obs_, action_, reward_, done_ = [], [], [], [], []

    for i in range(N - 1):
        obs_.append(dataset['observations'][i].astype(np.float32))
        next_obs_.append(dataset['observations'][i + 1].astype(np.float32))
        action_.append(dataset['actions'][i].astype(np.float32))
        reward_.append(dataset['rewards'][i].astype(np.float32))
        done_.append(dataset['timeouts'][i].astype(np.float32))

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


def process_maze2d_datasets():
    """
    Processes Maze2D datasets and saves them to disk.
    """
    env_names = [
        'maze2d-umaze-v1',
        'maze2d-medium-v1',
        'maze2d-large-v1',
    ]
    info = {}
    for env_name in env_names:
        print(f"Processing {env_name}:")
        env = gym.make(env_name)
        dataset = customize_qlearning_dataset(env)
        env.close()

        dones_float = np.zeros_like(dataset['rewards'])
        for i in range(len(dones_float) - 1):
            if (np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6 or
                    dataset['terminals'][i] == 1.0):
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        dones_float[-1] = 1

        trajectories = split_into_trajectories(
            dones_float,
            dataset['observations'],
            dataset['next_observations'],
            dataset['actions'],
            dataset['rewards'],
        )

        returns = np.array([np.sum(traj['rewards']) for traj in trajectories])
        lengths = np.array([len(traj['rewards']) for traj in trajectories])
        num_samples = np.sum(lengths)
        num_nonzero_tot_rew = np.sum(returns != 0)
        print(f"Number of samples collected: {num_samples}")
        print(f"Number of trajectories: {len(trajectories)}")
        print(f"Number of non-zero return trajectories: {num_nonzero_tot_rew}")
        print(f"Trajectory returns: mean={np.mean(returns):.2f}, std={np.std(returns):.2f}, "
              f"max={np.max(returns):.2f}, min={np.min(returns):.2f}")
        print(f"Trajectory lengths: mean={np.mean(lengths):.2f}, std={np.std(lengths):.2f}, "
              f"max={np.max(lengths)}, min={np.min(lengths)}")
        print("-" * 30)

        # Prepare dataset for saving
        dataset_list = [{
            'observations': traj['observations'],
            'actions': traj['actions'],
            'rewards': traj['rewards'],
            'dones': traj['terminals']
        } for traj in trajectories]

        dataset_dict = {key: [d[key] for d in dataset_list] for key in dataset_list[0]}
        dataset_hf = Dataset.from_dict(dataset_dict)
        dataset_hf = DatasetDict({'train': dataset_hf})
        print(dataset_hf)

        # Save dataset to disk
        directory_path = f'{env_name}'
        os.makedirs(directory_path, exist_ok=True)
        dataset_hf.save_to_disk(directory_path)

        # Compute mean and std of states
        state_mean, state_std = get_dataset_mean_std(dataset_hf['train'])
        info[env_name] = {'state_mean': state_mean, 'state_std': state_std}

    print(info)


def process_kitchen_datasets():
    """
    Processes Kitchen datasets and saves them to disk.
    """
    env_names = ['kitchen-complete-v0', 'kitchen-partial-v0', 'kitchen-mixed-v0']
    info = {}
    for env_name in env_names:
        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)
        env.close()

        dones_float = np.zeros_like(dataset['rewards'])
        for i in range(len(dones_float) - 1):
            dones_float[i] = 1.0 if dataset['terminals'][i] == 1.0 else 0.0
        dones_float[-1] = 1

        trajectories = split_into_trajectories(
            dones_float,
            dataset['observations'],
            dataset['next_observations'],
            dataset['actions'],
            dataset['rewards'],
        )

        print(f"Processing {env_name}:")
        returns = np.array([np.sum(traj['rewards']) for traj in trajectories])
        lengths = np.array([len(traj['rewards']) for traj in trajectories])
        num_samples = np.sum(lengths)
        print(f"Number of samples collected: {num_samples}")
        print(f"Number of trajectories: {len(trajectories)}")
        print(f"Trajectory returns: mean={np.mean(returns):.2f}, std={np.std(returns):.2f}, "
              f"max={np.max(returns):.2f}, min={np.min(returns):.2f}")
        print(f"Trajectory lengths: mean={np.mean(lengths):.2f}, std={np.std(lengths):.2f}, "
              f"max={np.max(lengths)}, min={np.min(lengths)}")

        # Prepare dataset for saving
        dataset_list = [{
            'observations': traj['observations'],
            'actions': traj['actions'],
            'rewards': traj['rewards'],
            'dones': traj['terminals']
        } for traj in trajectories]

        dataset_dict = {key: [d[key] for d in dataset_list] for key in dataset_list[0]}
        dataset_hf = Dataset.from_dict(dataset_dict)
        dataset_hf = DatasetDict({'train': dataset_hf})
        print(dataset_hf)

        # Save dataset to disk
        directory_path = f'{env_name}'
        os.makedirs(directory_path, exist_ok=True)
        dataset_hf.save_to_disk(directory_path)

        # Compute mean and std of states
        state_mean, state_std = get_dataset_mean_std(dataset_hf['train'])
        info[env_name] = {'state_mean': state_mean, 'state_std': state_std}

    print(info)



if __name__ == '__main__':
    process_mujoco_datasets()
    process_antmaze_datasets()
    process_maze2d_datasets()
    process_kitchen_datasets()

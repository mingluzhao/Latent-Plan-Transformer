import argparse

# Environment-specific parameters
ENV_CONFIG = {
    # Gym-Mujoco Environments
    'halfcheetah': {
        'n_layer': 3,
        'n_head': 1,
        'hidden_size': 128,
        'context_len': 32,
        'learning_rate': 1e-4,
        'langevin_step_size': 0.3,
        'nonlinearity': 'ReLU',
        'env_targets': 6000,
        'max_len': 1000,
        'scale': 1000.0
    },
    'walker2d': {
        'n_layer': 3,
        'n_head': 1,
        'hidden_size': 128,
        'context_len': 64,
        'learning_rate': 1e-4,
        'langevin_step_size': 0.3,
        'nonlinearity': 'ReLU',
        'env_targets': 5000,
        'max_len': 1000,
        'scale': 1000.0
    },
    'hopper': {
        'n_layer': 3,
        'n_head': 1,
        'hidden_size': 128,
        'context_len': 64,
        'learning_rate': 1e-4,
        'langevin_step_size': 0.3,
        'nonlinearity': 'ReLU',
        'env_targets': 3600,
        'max_len': 1000,
        'scale': 1000.0
    },
    'antmaze-umaze': {
        'n_layer': 3,
        'n_head': 1,
        'hidden_size': 192,
        'context_len': 64,
        'learning_rate': 1e-3,
        'langevin_step_size': 0.3,
        'nonlinearity': 'ReLU',
        'env_targets': 1,
        'max_len': 700,  # 700 for antmaze-umaze, 1000 otherwise
        'scale': 1.0
    },
    # Maze2D Environments
    'maze2d-umaze': {
        'n_layer': 1,
        'n_head': 8,
        'hidden_size': 128,
        'context_len': 32,
        'learning_rate': 1e-3,
        'langevin_step_size': 0.3,
        'nonlinearity': 'ReLU',
        'env_targets': 165,
        'max_len': 300,
        'scale': 100.0
    },
    'maze2d-medium': {
        'n_layer': 3,
        'n_head': 1,
        'hidden_size': 192,
        'context_len': 64,
        'learning_rate': 1e-3,
        'langevin_step_size': 0.3,
        'nonlinearity': 'ReLU',
        'env_targets': 280,
        'max_len': 600,
        'scale': 100.0
    },
    'maze2d-large': {
        'n_layer': 4,
        'n_head': 4,
        'hidden_size': 192,
        'context_len': 64,
        'learning_rate': 2e-4,
        'langevin_step_size': 0.3,
        'nonlinearity': 'ReLU',
        'env_targets': 275,
        'max_len': 800,
        'scale': 100.0
    },
    # Franka Kitchen Environments
    'kitchen-mixed': {
        'n_layer': 4,
        'n_head': 4,
        'hidden_size': 128,
        'context_len': 16,
        'learning_rate': 1e-3,
        'langevin_step_size': 0.3,
        'nonlinearity': 'ReLU',
        'env_targets': 4,
        'max_len': 280,
        'scale': 1.0
    },
    'kitchen-partial': {
        'n_layer': 3,
        'n_head': 16,
        'hidden_size': 128,
        'context_len': 16,
        'learning_rate': 1e-3,
        'langevin_step_size': 0.3,
        'nonlinearity': 'ReLU',
        'env_targets': 4,
        'max_len': 280,
        'scale': 1.0
    },
}

def get_args():
    parser = argparse.ArgumentParser(description='Training settings')

    # Environment and dataset settings
    parser.add_argument('--env_name', '-e', type=str, default='halfcheetah')
    parser.add_argument('--eval_dataset', type=str, default='medium-replay')  # Options: medium / medium-replay

    # Training parameters (default values, can be overridden by ENV_CONFIG)
    parser.add_argument('--attn', type=int, default=None)
    parser.add_argument('--max_len', type=int, default=None)  
    parser.add_argument('--langevin_step_size', type=float, default=None)  
    parser.add_argument('--reward_weight', type=float, default=0.25)
    parser.add_argument('--hidden_size', type=int, default=None)  
    parser.add_argument('--nlayer', type=int, default=None)  
    parser.add_argument('--nhead', type=int, default=None)  
    parser.add_argument('--nonlinearity', type=str, default=None)  
    parser.add_argument('--lr', type=float, default=None)  

    # Other training parameters
    parser.add_argument('--sample_by_length', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--latent', type=int, default=4)
    parser.add_argument('--omega_cg', type=int, default=1)

    args = parser.parse_args()

    # Load environment-specific parameters
    env_key = args.env_name.lower()
    # Handle special cases where env_name might contain additional info (e.g., '-v0')
    for key in ENV_CONFIG.keys():
        if key in env_key:
            env_params = ENV_CONFIG[key]
            break
    else:
        raise ValueError(f"Environment '{args.env_name}' not recognized in ENV_CONFIG.")

    # Override default arguments with environment-specific parameters
    args.nlayer = env_params['n_layer']
    args.nhead = env_params['n_head']
    args.hidden_size = env_params['hidden_size']
    args.attn = env_params['context_len']
    args.lr = env_params['learning_rate']
    args.langevin_step_size = env_params['langevin_step_size']
    args.nonlinearity = env_params['nonlinearity']
    args.env_targets = env_params['env_targets']
    args.max_len = env_params['max_len']
    args.scale = env_params['scale']

    return args

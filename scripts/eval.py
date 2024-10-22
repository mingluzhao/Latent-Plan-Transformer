import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
from model.utils_eval import evaluate_env
from model.lpt import LPT, DecisionTransformerConfig
from config import ENV_CONFIG

def main():
    """
    Main function for evaluating a trained model on a specified environment.
    """
    parser = argparse.ArgumentParser(description='Evaluation settings')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for evaluation')
    parser.add_argument('--omega_cg', type=float, default=1.0, help='Coefficient for classifier guidance')
    parser.add_argument('--file_dir', type=str, required=True, help='Path to the trained model directory')
    parser.add_argument('--env_name', type=str, required=True, help='Name of the environment')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Retrieve environment parameters from configuration
    env_key = args.env_name.lower()
    env_params = None
    for key in ENV_CONFIG.keys():
        if key in env_key:
            env_params = ENV_CONFIG[key]
            break
    if env_params is None:
        raise ValueError(f"Environment '{args.env_name}' not recognized in ENV_CONFIG.")

    # Override default arguments with environment-specific parameters
    args.n_layer = env_params['n_layer']
    args.n_head = env_params['n_head']
    args.hidden_size = env_params['hidden_size']
    args.context_len = env_params['context_len']
    args.learning_rate = env_params['learning_rate']
    args.langevin_step_size = env_params['langevin_step_size']
    args.nonlinearity = env_params['nonlinearity']
    args.env_targets = env_params['env_targets']
    args.max_len = env_params['max_len']
    args.scale = env_params['scale']

    # Determine evaluation dataset
    eval_dataset = 'medium-replay' if 'replay' in args.file_dir else 'medium'

    # Load model configuration and initialize model
    config_path = os.path.join(args.file_dir, 'config.json')
    config = DecisionTransformerConfig.from_pretrained(config_path)
    model = LPT(config)
    model = model.from_pretrained(args.file_dir)

    # Set evaluation device
    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate the model
    metrics = evaluate_env(
        model=model,
        device=eval_device,
        env_name=args.env_name,
        eval_dataset=eval_dataset,
        env_targets=args.env_targets,
        scale=args.scale,
        num_eval_ep=50,
        seed=args.seed,
        omega_cg=args.omega_cg
    )

    # Print evaluation results
    print({
        'env': args.env_name,
        'dataset': eval_dataset,
        'seed': args.seed,
        'omega_cg': args.omega_cg,
        'target': args.env_targets,
        'avg_reward': metrics['eval/avg_reward'],
        'avg_ep_len': metrics['eval/avg_ep_len'],
        'norm_score': metrics['eval/norm_score'],
        'file': args.file_dir
    })

if __name__ == "__main__":
    main()

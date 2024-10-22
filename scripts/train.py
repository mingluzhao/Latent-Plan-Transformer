import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lpt import DecisionTransformerConfig, DecisionTransformerModel, LPT
from model.utils_train import LPTGymDataCollator
from model.utils_eval import evaluate_env

import numpy as np
import torch
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, TrainerCallback
from config import get_args

def main():
    args = get_args()

    # Set random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    env_name = args.env_name
    eval_dataset = args.eval_dataset
    window_size = args.attn
    langevin_step_size = args.langevin_step_size
    reward_weight = args.reward_weight
    lr = args.lr
    hidden_size = args.hidden_size
    sample_by_length = args.sample_by_length
    num_train_epochs = args.epochs
    batch_size = args.batch
    n_layer = args.nlayer
    n_head = args.nhead
    n_latent = args.latent
    omega_cg = args.omega_cg
    nonlinearity = args.nonlinearity
    env_targets = args.env_targets
    max_len = args.max_len
    scale = args.scale

    # Load dataset based on environment name
    if "antmaze" in env_name:
        directory_path = f'../data/dataset/{env_name}-v2'
    elif "maze2d" in env_name:
        directory_path = f'../data/dataset/{env_name}-v1'
    elif 'kitchen' in env_name:
        directory_path = f'../data/dataset/{env_name}-v0'
    else:
        directory_path = f'../data/dataset/{env_name}-{eval_dataset}-v2'

    dataset = load_from_disk(directory_path)

    # Initialize data collator and configuration
    collator = LPTGymDataCollator(
        dataset["train"],
        max_len=max_len,
        scale=scale,
        sample_by_length=sample_by_length,
    )

    config = DecisionTransformerConfig(
        state_dim=collator.state_dim,
        act_dim=collator.act_dim,
        n_traj=collator.n_traj,
        add_cross_attention=True,
        max_ep_len=max_len,
        window_size=window_size,
        langevin_step_size=langevin_step_size,
        reward_weight=reward_weight,
        hidden_size=hidden_size,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        n_layer=n_layer,
        n_head=n_head,
        n_latent=n_latent,
        omega_cg=omega_cg,
        nonlinearity=nonlinearity,
    )

    model = LPT(config)

    # Define output directory
    output_dir = os.path.join(
        '../output', env_name,
        f'attn{window_size}_lr{lr}_hidden{hidden_size}_layer{n_layer}_head{n_head}'
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        remove_unused_columns=False,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        optim="adamw_torch",
        max_grad_norm=0.25,
        report_to=[],  # No reporting to external services
    )

    eval_device = "cuda" if torch.cuda.is_available() else "cpu"

    def evaluate(model):
        return evaluate_env(
            model=model,
            device=eval_device,
            env_name=env_name,
            eval_dataset=eval_dataset,
            env_targets=env_targets,
            scale=scale,
            num_eval_ep=10,
            omega_cg=omega_cg
        )

    # Define a simple callback for evaluation and model saving
    class EvalCallback(TrainerCallback):
        def __init__(self, eval_func, eval_steps, output_dir):
            self.eval_func = eval_func
            self.eval_steps = eval_steps
            self.output_dir = output_dir

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.eval_steps == 0 and state.global_step > 0:
                metrics = self.eval_func(kwargs['model'])
                print(f"Evaluation metrics at step {state.global_step}: {metrics}")
                save_path = os.path.join(self.output_dir, f"model_{state.global_step}")
                kwargs['model'].save_pretrained(save_path)

    # Initialize the trainer with the evaluation callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=collator,
        callbacks=[EvalCallback(evaluate, eval_steps=500, output_dir=output_dir)]
    )

    trainer.train()


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##########################################################################################################
########################################## LAUCH SIMULATION ##############################################
##########################################################################################################
"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num-envs", type=int, default=10, help="Number of environments to spawn.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

##########################################################################################################
############################################# RL SETUP ###################################################
##########################################################################################################
from typing import Dict, Tuple
import torch
import shutil
from pathlib import Path
import numpy as np
import torch.nn as nn
import gymnasium as gym

from franka_env import ManagerRLGoalEnv, FrankaShadowLiftEnvCfg

from drl.utils.env_utils import IsaacVecEnv
from drl.agent.sac import CSAC_GCRL
from drl.memory.rher import RHERMemory
from drl.learning.rher import RHER
from drl.utils.optim.adamw import AdamWOptimizer
from drl.utils.nn.seq import SeqGRUNet

ENV_CFG = FrankaShadowLiftEnvCfg()


class PolicyNetwork(nn.Module):
    def __init__(
        self, 
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Box
    ):
        super().__init__()
        self.obs_dim = observation_space['observation'].shape[0]
        self.n_tasks, self.goal_dim = observation_space['desired_goal'].shape
        self.act_dim = action_space.shape[0]
        
        self.net = SeqGRUNet(
            obs_dim=self.obs_dim, 
            meta_dim=self.goal_dim,
            out_dim=self.n_tasks * 2 * self.act_dim,
            embed_dim=256, num_layers=1,
            hidden_mlp_dims=[1024, 768, 512],
            use_norm=True, activation='SiLU' 
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        (obs, mask), meta = x['observation'], x['goal']

        B, device = len(meta), meta.device
        batch_idxs = torch.arange(B, device=device)
        task_idxs = x.get('task_id', torch.full((B,), -1, dtype=torch.long, device=device))  # (B,)

        return self.net(obs, meta, mask).view(B, self.n_tasks, 2 * self.act_dim)[batch_idxs, task_idxs].chunk(2, dim=1) # (mu, std)
    

class ValueNetwork(nn.Module):
    def __init__(
        self, 
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Box
    ):
        super().__init__()
        self.obs_dim = observation_space['observation'].shape[0]
        self.n_tasks, self.goal_dim = observation_space['desired_goal'].shape
        self.act_dim = action_space.shape[0]
        
        self.net = SeqGRUNet(
            obs_dim=self.obs_dim, 
            meta_dim=self.goal_dim + self.act_dim,
            out_dim=self.n_tasks,
            embed_dim=256, num_layers=1,
            hidden_mlp_dims=[1024, 768, 512],
            use_norm=True, activation='SiLU' 
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        (obs, mask), meta = x['observation'], torch.cat([x['goal'], x['action']], dim=-1)

        B, device = len(meta), meta.device
        batch_idxs = torch.arange(B, device=device)
        task_idxs = x.get('task_id', torch.full((B,), -1, dtype=torch.long, device=device))  # (B,)

        return self.net(obs, meta, mask)[batch_idxs, task_idxs].view(-1, 1)


def make_optimizer_fn(params, model_name: str, lr_scale: float = 1.0):
    """
    Create optimizer with learning rate scaled by num_envs.
    
    Args:
        params: Model parameters
        model_name: 'actor', 'critic', or 'coef'
        lr_scale: Learning rate scaling factor (e.g., sqrt(num_envs / 10))
    """
    assert model_name in ['actor', 'critic', 'coef'], ValueError(model_name)
    
    # Base learning rate (tuned for num_envs=10, batch_size=512)
    base_lr = 3e-4
    scaled_lr = base_lr * lr_scale
    
    return AdamWOptimizer(
        params=params, polyak=5e-3,
        lr=scaled_lr, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.0
    )
    

if __name__ == '__main__':
    # Set seed for reproducibility
    import random
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    
    # Optimize PyTorch for multi-environment training
    torch.set_num_threads(4)  # Limit CPU threads to avoid contention
    torch.backends.cudnn.benchmark = True  # Auto-tune CUDA kernels
    
    # ============================================================================
    # ADAPTIVE SCALING: Automatically adjust parameters based on num_envs
    # ============================================================================
    BASE_NUM_ENVS = 10  # Baseline configuration
    env_scale_factor = args_cli.num_envs / BASE_NUM_ENVS
    
    # Scale batch_size and num_updates proportionally to num_envs
    # This ensures consistent sample-to-update ratio
    base_batch_size = 512
    base_num_updates = 128
    scaled_batch_size = int(base_batch_size * env_scale_factor)
    scaled_num_updates = int(base_num_updates * env_scale_factor)
    
    # Scale replay buffer to hold enough experiences
    # Maintain ~10 cycles worth of data regardless of num_envs
    base_memory_size = 20000
    scaled_memory_size = int(base_memory_size * env_scale_factor)
    
    # CRITICAL: Scale learning rate to compensate for larger batch sizes
    # Use square root scaling (common practice in distributed training)
    # num_envs=10: lr_scale=1.0, num_envs=40: lr_scale=2.0, num_envs=100: lr_scale=3.16
    lr_scale = np.sqrt(env_scale_factor)
    scaled_lr = 3e-4 * lr_scale
    
    scaled_num_cycles = 200  
    
    # CRITICAL FIX: num_cycles MUST be divisible by num_envs (RHER requirement)
    # Round up to nearest multiple of num_envs
    scaled_num_cycles = ((scaled_num_cycles + args_cli.num_envs - 1) // args_cli.num_envs) * args_cli.num_envs
    
    print("=" * 80)
    print("ADAPTIVE SCALING CONFIGURATION")
    print("=" * 80)
    print(f"Number of environments:  {args_cli.num_envs} (base: {BASE_NUM_ENVS})")
    print(f"Scale factor:            {env_scale_factor:.2f}x")
    print(f"Batch size:              {scaled_batch_size} (base: {base_batch_size})")
    print(f"Num updates per cycle:   {scaled_num_updates} (base: {base_num_updates})")
    print(f"Replay buffer size:      {scaled_memory_size} (base: {base_memory_size})")
    print(f"Learning rate:           {scaled_lr:.2e} (base: 3e-4, scale: {lr_scale:.2f}x)")
    print(f"Cycles per epoch:        {scaled_num_cycles} (FIXED at 200)")
    print(f"Total samples per epoch: {args_cli.num_envs * scaled_num_cycles * ENV_CFG.num_frames:,}")
    print(f"Speedup vs base:         {env_scale_factor:.1f}x more samples/epoch")
    print("=" * 80)
    print()
    
    # ============================================================================
    # CLEANUP OLD CHECKPOINTS: Keep only last 3 experiments to save disk space
    # ============================================================================
    runs_dir = Path("./runs")
    if runs_dir.exists():
        exp_folders = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("exp")],
                           key=lambda x: x.stat().st_mtime)  # Sort by modification time
        
        # Keep only the 3 most recent experiments
        MAX_EXPERIMENTS = 3
        if len(exp_folders) > MAX_EXPERIMENTS:
            folders_to_delete = exp_folders[:-MAX_EXPERIMENTS]
            print("=" * 80)
            print("CHECKPOINT CLEANUP")
            print("=" * 80)
            for folder in folders_to_delete:
                print(f"Removing old experiment: {folder.name}")
                shutil.rmtree(folder, ignore_errors=True)
            print(f"Kept {MAX_EXPERIMENTS} most recent experiments")
            print("=" * 80)
            print()
    
    # Create custom environment config class with seed
    class FrankaShadowLiftEnvCfgWithSeed(FrankaShadowLiftEnvCfg):
        def __init__(self):
            super().__init__()
            self.seed = args_cli.seed
    
    envs = IsaacVecEnv(
        manager=ManagerRLGoalEnv,
        cfg=FrankaShadowLiftEnvCfgWithSeed,
        num_envs=args_cli.num_envs
    )
    
    # Create optimizer factory with scaled learning rate
    def make_optimizer_fn_scaled(params, model_name: str):
        return make_optimizer_fn(params, model_name, lr_scale=lr_scale)
    
    agent = CSAC_GCRL(
        make_policy_fn=PolicyNetwork,
        make_value_fn=ValueNetwork,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        make_optimizer_fn=make_optimizer_fn_scaled,  # Use scaled version
        num_ent_coefs=ENV_CFG.num_stages, 
        device='cuda'
    )
    memory = RHERMemory(
        reward_func=ENV_CFG.reward_func,
        num_stages=ENV_CFG.num_stages,
        horizon=int(ENV_CFG.episode_length_s / (ENV_CFG.sim.dt * ENV_CFG.decimation)),
        max_length=scaled_memory_size,  # SCALED
        device='cuda'
    )
    learner = RHER(
        envs=envs,
        agent=agent,
        memory=memory
    )
    learner.run(
        epochs=2000,
        num_cycles=scaled_num_cycles,    # SCALED
        num_eval_episodes=100,
        r_mix=0.5,
        num_updates=scaled_num_updates,  # SCALED
        batch_size=scaled_batch_size,    # SCALED
        future_p=0.8,
        discounted_factor=0.98,
        clip_return=None,
        n_steps=ENV_CFG.num_frames,
        step_decay=0.7
    )
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


def make_optimizer_fn(params, model_name: str):
    assert model_name in ['actor', 'critic', 'coef'], ValueError(model_name)
    # Increased learning rate for faster initial exploration
    return AdamWOptimizer(
        params=params, polyak=5e-3,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.0
    )
    

if __name__ == '__main__':
    envs = IsaacVecEnv(
        manager=ManagerRLGoalEnv,
        cfg=FrankaShadowLiftEnvCfg,
        num_envs=args_cli.num_envs
    )
    agent = CSAC_GCRL(
        make_policy_fn=PolicyNetwork,
        make_value_fn=ValueNetwork,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        make_optimizer_fn=make_optimizer_fn,
        num_ent_coefs=ENV_CFG.num_stages,
        device='cuda'
    )
    memory = RHERMemory(
        reward_func=ENV_CFG.reward_func,
        num_stages=ENV_CFG.num_stages,
        horizon=int(ENV_CFG.episode_length_s / (ENV_CFG.sim.dt * ENV_CFG.decimation)),
        max_length=20000,
        device='cuda'
    )
    learner = RHER(
        envs=envs,
        agent=agent,
        memory=memory
    )
    learner.run(
        epochs=2000,
        num_cycles=200,        
        num_eval_episodes=50,  
        r_mix=0.5,             
        num_updates=128,       
        batch_size=512,
        future_p=0.8,                   
        discounted_factor=0.98,
        clip_return=None,
        n_steps=ENV_CFG.num_frames,
        step_decay=0.7,
        resume_path="C:\\Users\\PC\\Downloads\\KLTN_AnhChinh\\KLTN_Robotics\\runs\\exp3"
    )
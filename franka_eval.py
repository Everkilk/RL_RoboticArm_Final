# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Disable __pycache__ creation
import sys
sys.dont_write_bytecode = True

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
import time
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym

from franka_env import ManagerRLGoalEnv, FrankaShadowLiftEnvCfg

from drl.utils.env_utils import IsaacEnv
from drl.agent.sac import CSAC_GCRL
from drl.memory.rher import RHERMemory
from drl.learning.rher import RHER
from drl.utils.optim.adamw import AdamWOptimizer
from drl.utils.nn.seq import SeqGRUNet
from drl.utils.general import map_structure

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
    


if __name__ == '__main__':
    env = IsaacEnv(
        manager=ManagerRLGoalEnv,
        cfg=FrankaShadowLiftEnvCfg
    )
    policy = PolicyNetwork(
        observation_space=env.observation_space,
        action_space=env.action_space
    )
    # print(env.action_space.shape)
    # input()
    policy_state_dict = torch.load(r'runs\best_policy\lift\best_policy.pt', map_location='cpu', weights_only=False)
    print('Policy Loading: ', policy.load_state_dict(policy_state_dict))
    policy = policy.eval().cuda()

    for _ in range(30):
        obs = env.reset()[0]
        done = False
        while not done:

            with torch.no_grad():
                input_dict = {
                    'observation': map_structure(lambda x: torch.from_numpy(x).unsqueeze(0).cuda(), obs['observation']),
                    'goal': torch.from_numpy(obs['desired_goal'][-1]).unsqueeze(0).cuda()
                }
                action = policy(input_dict)[0].squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            time.sleep(0.05)
from typing import Union, Callable, Optional, Dict, Any

import torch
import numpy as np
from gymnasium import spaces

from drl.agent.base import Agent
from drl.utils.model.deterministics import CDetPolicy
from drl.utils.model.values import DeepValueFunc
from drl.utils.optim.base import Optimizer
from drl.utils.functional import mean_square_error, symmetric_info_nce, reduce_by_mean
from drl.utils.general import map_structure, Params, Device

###################################################################################################################################################
################################ DEEP DETERMINISTIC POLICY GRADIENT FOR GOAL-CONDITIONED REINFORCEMENT LEARNING  ##################################
###################################################################################################################################################
class DDPG_GCRL(Agent):
    """ Deep Deterministic Policy Gradient algorithm for Goal-conditioned RL continuous action space. """
    contrastive: bool = False
    def __init__(
        self, 
        make_policy_fn: Callable[[Dict[str, Any]], torch.nn.Module], 
        make_value_fn: Callable[[Dict[str, Any]], torch.nn.Module], 
        observation_space: spaces.Space, 
        action_space: spaces.Box,
        make_optimizer_fn: Callable[[Params, str], Optimizer],
        *,
        act_noise: float = 0.1,
        device: Device = None
    ):
        # check spaces and parameters --------------------------------
        assert isinstance(action_space, spaces.Box), TypeError(f'Invalid action_space ({action_space}).')
        assert act_noise >= 0, ValueError(f'Invalid act_noise ({act_noise}).')
        self.act_noise = act_noise

        # build agent ------------------------------------------------
        super().__init__(
            observation_space=observation_space,
            action_space=action_space, 
            target_policy_path='actor.policy_net',
            models={
                'actor': CDetPolicy(
                    make_policy_fn=make_policy_fn,
                    observation_space=observation_space,
                    action_space=action_space,
                    use_target=True
                ),
                'critic': DeepValueFunc(
                    value_type='q',
                    make_value_fn=make_value_fn,
                    observation_space=observation_space,
                    action_space=action_space,
                    use_target=True
                )
            },
            make_optimizer_fn=make_optimizer_fn,
            device=device, coefs=None    
        )
    
    # select action methods ---------------------------------------------
    def forward(self, input_dict: Dict[str, Any], deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.actor.select_best(input_dict)
        return self.actor.sample(input_dict, act_noise=self.act_noise)
    
    # update methods ----------------------------------------------------    
    def backward(
        self, 
        data: Dict[str, Any], 
        discounted_factor: float = 0.99,
        clip_return: Optional[float] = None
    ) -> Dict[str, float]:
        # extract data -------------------------------------------------
        observations = data['observations']
        actions = data['actions']
        next_observations = data['next_observations']
        goals = data['goals']
        rewards = data['rewards'] 
        terminals = data['terminals']
        task_ids = data.get('task_ids', None)
        step_decays = data.get('step_decays', None)
        
        # update critic ------------------------------------------------
        with torch.no_grad():
            if step_decays is not None:
                n_steps = step_decays.size(1)
                seq_discount_rate = discounted_factor**torch.arange(n_steps, device=self.device)
                # compute Q value for each next steps
                step_q_nexts = []
                for step_id in range(n_steps):
                    # format data
                    next_input_dict = {
                        'observation': map_structure(lambda x: x[:, step_id], next_observations),
                        'goal': goals
                    }
                    if task_ids is not None:
                        next_input_dict['task_id'] = task_ids
                    # compute Q next
                    next_input_dict['action'] = self.actor(next_input_dict, return_target=True)
                    q_nexts = self.critic(next_input_dict, return_target=True)
                    # compute store step Q next
                    step_q_nexts.append(q_nexts)
                step_q_nexts = seq_discount_rate * torch.cat(step_q_nexts, dim=1)
                # compute Q target for each next steps
                q_actuals_n = torch.cumsum(seq_discount_rate * rewards, dim=1) + discounted_factor * (1 - terminals) * step_q_nexts
                # compute final Q target, using exponential decay weights
                step_decay_weights = step_decays / torch.sum(step_decays, dim=1, keepdim=True)
                q_actuals = torch.sum(step_decay_weights * q_actuals_n, dim=-1, keepdim=True)
            else:
                # format data
                next_input_dict = {
                    'observation': next_observations,
                    'goal': goals
                }
                if task_ids is not None:
                    next_input_dict['task_id'] = task_ids
                # compute Q next
                next_input_dict['action'] = self.actor(next_input_dict, return_target=True)
                q_nexts = self.critic(next_input_dict, return_target=True)

                # compute Q target
                q_actuals = rewards + discounted_factor * (1 - terminals) * q_nexts
            
            if clip_return is not None:
                q_actuals = torch.clamp(q_actuals, -clip_return, clip_return)

        # forward
        input_dict = {
            'observation': observations,
            'goal': goals,
            'action': actions
        }
        if task_ids is not None:
            input_dict['task_id'] = task_ids
        # compte Q losses
        q_losses = mean_square_error(self.critic(input_dict), q_actuals)
        # update params
        critic_loss = self.critic_optim.backward(reduce_by_mean(q_losses))

        # update actor -------------------------------------------------
        input_dict = {
            'observation': observations,
            'goal': goals
        }
        if task_ids is not None:
            input_dict['task_id'] = task_ids
        input_dict['action'] = self.actor(input_dict)
        # compute policy losses
        p_losses = -self.critic(input_dict)
        # update params
        actor_loss = self.actor_optim.backward(reduce_by_mean(p_losses))

        # return evaluations and compute td errors --------------------
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
        }
    
    # estimate values / evaluate actions ----------------------------------------
    @torch.inference_mode()
    def compute_action_value(self, input_dict: Dict[str, Any], actions: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        # format data
        input_dict = map_structure(self.format_data, input_dict)
        input_dict['action'] = self.format_data(actions)
        # compute Q-values
        return self.critic(input_dict)
    
    @torch.inference_mode()
    def compute_state_value(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        # format data
        input_dict = map_structure(self.format_data, input_dict)
        input_dict['action'] = self.actor(input_dict)
        # compute V-values
        return self.compute_action_value(input_dict)
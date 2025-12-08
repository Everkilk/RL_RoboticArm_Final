from typing import Tuple, Union, Callable, Optional, Any, Dict

import torch
import numpy as np
from gymnasium import spaces

from drl.agent.base import Agent
from drl.utils.optim.base import Optimizer

from drl.utils.model.stochastics import TanhNormalPolicy
from drl.utils.model.values import EnsembleDeepValueFunc
from drl.utils.model.coefficients import Coefficients

from drl.utils.functional import mean_square_error, clip_coef, reduce_by_mean
from drl.utils.general import map_structure, Device, Params

###########################################################################################################
######################### CONTINUOUS SOFT ACTOR-CRITIC + GOAL-CONDITIONED RL ##############################
###########################################################################################################
class SAC_GCRL(Agent):
    """ Continuous Soft Actor-Critic with Goal-Conditioned Reinforcement Learning. """
    def __init__(
        self, 
        make_policy_fn: Callable[[spaces.Space, spaces.Space], torch.nn.Module],
        make_value_fn: Callable[[spaces.Space, spaces.Space], torch.nn.Module], 
        observation_space: spaces.Dict, 
        action_space: spaces.Box, 
        make_optimizer_fn: Callable[[Params, str], Optimizer],
        log_std_range: Tuple[float, float] = (-6.0, 2.0),
        num_ensembles: int = 2, 
        num_subsets: Optional[int] = None,
        num_ent_coefs: int = 1,
        ent_target_scale: float = 1.0,
        ent_coef_range: Tuple[float, float] = (0.0, 1.0),
        init_ent_coef: float = 1.0,
        device: Device = None
    ):
        # check spaces and parameters ---------------------------------
        assert isinstance(observation_space, spaces.Dict), TypeError(f'Invalid observation_space type ({type(observation_space)}).')
        for obs_name in ['observation', 'achieved_goal', 'desired_goal']:
            assert obs_name in observation_space.keys(), KeyError(f'{obs_name} is not in observation_space.')
        assert isinstance(action_space, spaces.Box), TypeError(f'Invalid action_space ({action_space}).')
        assert num_ensembles > 1, ValueError(f'Invalid num_ensembles ({num_ensembles}).')
        self.redq = (num_ensembles > 2) and (num_subsets is not None)
        if self.redq:
            assert num_subsets > 2, ValueError(f'num_subsets ({num_subsets}) must be larger than 2 in REQD mode.')
            assert num_subsets < num_ensembles, ValueError(f'num_subsets ({num_subsets}) must be less than num_ensembles ({num_ensembles}).')
        self.num_subsets = num_subsets

        # build agent --------------------------------------------------
        super().__init__(
            observation_space=observation_space,
            action_space=action_space, 
            target_policy_path='actor.policy_net',
            models={
                'actor': TanhNormalPolicy(
                    make_policy_fn=make_policy_fn,
                    observation_space=observation_space,
                    action_space=action_space,
                    log_std_range=log_std_range
                ),
                'critic': EnsembleDeepValueFunc(
                    value_type='q',
                    make_value_fn=make_value_fn,
                    observation_space=observation_space,
                    action_space=action_space,
                    num_ensembles=num_ensembles,
                    use_target=True
                ),
            },
            coefs={
                'ent_coef': Coefficients(
                    num_coefs=num_ent_coefs, 
                    target_value=ent_target_scale * action_space.shape[0], 
                    coef_range=ent_coef_range, 
                    init_coef=init_ent_coef
                )
            },
            make_optimizer_fn=make_optimizer_fn,
            device=device    
        )

    # select action ---------------------
    def forward(self, input_dict: Dict[str, Any], deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.actor.select_best(input_dict)
        return self.actor.sample(input_dict)

    # update models --------------------
    def backward(
        self, 
        data: Dict[str, Any], 
        discounted_factor: float = 0.99,
        clip_return: Union[float, None] = None
    ) -> Dict[str, float]:
        # extract data ---------------------------------------------------
        observations = data['observations']
        actions = data['actions']
        next_observations = data['next_observations']
        goals = data['goals']
        rewards = data['rewards'] # (batch_size, 1)
        terminals = data['terminals'] # (batch_size, 1)
        task_ids = data.get('task_ids', None)
        step_decays = data.get('step_decays', None)
        
        # get coefficients ---------------------------------------------
        with torch.no_grad():
            ent_coefs = self.ent_coef(task_ids).exp()
        
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
                    # compute action, log probs, and Q value
                    next_input_dict['action'], next_log_probs = self.actor(next_input_dict)[0]
                    ensemble_q_next_vals = self.critic(next_input_dict, num_subsets=self.num_subsets, return_target=True)
                    # compute Q next
                    step_q_nexts.append(ensemble_q_next_vals.min(dim=0)[0] - ent_coefs * next_log_probs)
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
                # compute action, log probs, and Q value
                next_input_dict['action'], next_log_probs = self.actor(next_input_dict)[0]
                ensemble_q_next_vals = self.critic(next_input_dict, num_subsets=self.num_subsets, return_target=True)
                # compute Q next
                q_nexts = ensemble_q_next_vals.min(dim=0)[0] - ent_coefs * next_log_probs
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
        ensemble_q_vals = self.critic(input_dict)
        # compte Q losses
        q_losses = mean_square_error(ensemble_q_vals, q_actuals).sum(dim=0)
        # update params
        critic_loss = self.critic_optim.backward(reduce_by_mean(q_losses))

        # update actor -------------------------------------------------
        input_dict = {
            'observation': observations,
            'goal': goals
        }
        if task_ids is not None:
            input_dict['task_id'] = task_ids
        input_dict['action'], log_probs = self.actor(input_dict)[0]
        ensemble_q_vals = self.critic(input_dict)
        q_vals = ensemble_q_vals.mean(dim=0) if self.redq else ensemble_q_vals.min(dim=0)[0]
        # compute policy losses
        p_losses = ent_coefs * log_probs - q_vals
        # update params
        actor_loss = self.actor_optim.backward(reduce_by_mean(p_losses))

        # tunning coefficients -----------------------------------------
        entropies = -log_probs.detach() # approximated entropies
        reg_ent_losses = clip_coef(
            coefs=self.ent_coef(task_ids, clip_coef=False), 
            deltas=-entropies.detach() - self.ent_coef.target_value, 
            coef_range=self.ent_coef.log_coef_range
        )
        _ = self.coef_optim.backward(reduce_by_mean(reg_ent_losses))

        # return evaluations and compute td errors --------------------
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'approx_ent': entropies.mean().item(),
            'ent_coef': ent_coefs.mean().item(),
        }
    
    # estimate values / evaluate actions ---------------------------------
    @torch.inference_mode()
    def compute_action_value(self, input_dict: Dict[str, Any], actions: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        # format data
        input_dict = map_structure(self.format_data, input_dict)
        input_dict['action'] = self.format_data(actions)
        # compute Q-values
        ensemble_q_vals = self.critic(input_dict)
        return ensemble_q_vals.mean(dim=0) if self.redq else ensemble_q_vals.amin(dim=0)
    
    @torch.inference_mode()
    def compute_state_value(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        # format data
        input_dict = map_structure(self.format_data, input_dict)
        # compute V-values
        actions, log_probs = self.actor(input_dict)[0]
        return self.compute_action_value(input_dict, actions) - self.ent_coef(input_dict.get('task_id', None)).exp() * log_probs

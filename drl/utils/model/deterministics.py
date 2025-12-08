from typing import Callable, Iterable, Optional, Dict, Any

from gymnasium import spaces

import torch
from torch import nn


###############################################################################################
################################### BASE DETERMINISTIC POLICY #################################
###############################################################################################
class DetPolicyBase(nn.Module):
    def __init__(
        self, 
        make_policy_fn: Callable[[Any], nn.Module],
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        use_target: bool = True
    ):
        super().__init__()
        # build source neural network
        self.policy_net = make_policy_fn(
            observation_space=observation_space,
            action_space=action_space
        )
        # build target neural network
        self.use_target = use_target
        if self.use_target:
            self.target_policy_net = make_policy_fn(
                observation_space=observation_space,
                action_space=action_space
            )
            self.target_policy_net.load_state_dict(self.policy_net.state_dict())
            self.target_policy_net.requires_grad_(False)
    
    # call methods -------------------------------------------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    # select action strategy methods --------------------------------------
    @torch.no_grad()
    def sample(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        """ Sample actions from the policy. """
        raise NotImplementedError
    
    # utility methods ---------------------------------------------------------
    def parameters(self) -> Iterable[nn.Parameter]:
        """ Return all parameters of the value function. """
        if self.use_target:
            return {
                'source': self.policy_net.parameters(),
                'target': self.target_policy_net.parameters()
            }
        return self.policy_net.parameters()
    

###############################################################################################
################################# DISCRETE DETERMINISTIC POLICY ###############################
###############################################################################################
class DDetPolicy(DetPolicyBase):
    """ Discrete Deterministic Policy for choosing deterministic actions. """
    def __init__(
        self, 
        make_policy_fn: Callable[[Any], nn.Module],
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        use_target: bool = True
    ):
        super().__init__(
            make_policy_fn=make_policy_fn,
            action_space=action_space,
            observation_space=observation_space,
            use_target=use_target
        )
        # register buffers --------------------------------
        self.register_buffer('start_action', torch.as_tensor(data=action_space.start, dtype=torch.long))

    # call methods -------------------------------------------------------------
    def forward(
        self, 
        input_dict: Dict[str, Any], 
        *, 
        return_target: bool = False
    ) -> torch.Tensor:
        """ Forward method to compute action. """
        if return_target:
            assert self.use_target, NotImplementedError('Model does not have target network')
            return self.target_policy_net(input_dict)
        return self.policy_net(input_dict)

    # select action strategy methods --------------------------------------
    @torch.no_grad()
    def sample(self, input_dict: Dict[str, Any], *, eps: float = 0.05) -> torch.Tensor:
        """ Sample actions from the policy. """
        # compute Q values
        q_vals = self(input_dict)
        # epsilon-greedy sample actions
        actions = q_vals.argmax(dim=-1)
        random_mask = torch.rand(q_vals.size(0), device=q_vals.device) < eps
        if random_mask.any():
            actions[random_mask] = torch.randint(
                low=0, high=q_vals.size(-1), 
                size=(torch.count_nonzero(random_mask),), 
                device=actions.device
            )
        return actions + self.start_action
    
    @torch.no_grad()
    def select_best(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        """ Select the best action based on the policy. """
        return self(input_dict).argmax(dim=-1) + self.start_action



###############################################################################################
################################# CONTINUOUS DETERMINISTIC POLICY #############################
###############################################################################################
class CDetPolicy(DetPolicyBase):
    """ Continuous Deterministic Policy for choosing deterministic actions. """
    def __init__(
        self, 
        make_policy_fn: Callable[[Any], nn.Module],
        observation_space: spaces.Space,
        action_space: spaces.Box,
        use_target: bool = True
    ):
        super().__init__(
            make_policy_fn=make_policy_fn,
            observation_space=observation_space,
            action_space=action_space,
            use_target=use_target
        )

        # register buffers --------------------------------
        self.register_buffer('action_low', torch.as_tensor(data=action_space.low, dtype=torch.float32))
        self.register_buffer('action_high', torch.as_tensor(data=action_space.high, dtype=torch.float32))
        self.register_buffer('action_scale', torch.as_tensor(data=(action_space.high - action_space.low) / 2, dtype=torch.float32))
        self.register_buffer('action_bias', torch.as_tensor(data=(action_space.high + action_space.low) / 2, dtype=torch.float32))

    # call methods -------------------------------------------------------------
    def forward(
        self, 
        input_dict: Dict[str, Any], 
        *, 
        act_noise: float = 0.0, 
        clip_noise: Optional[float] = None, 
        return_target: bool = False
    ) -> torch.Tensor:
        """ Forward method to compute action. """
        if return_target:
            assert self.use_target, NotImplementedError('Model does not have target network')
            mu = self.target_policy_net(input_dict)
        else:
            mu = self.policy_net(input_dict)
        actions = torch.tanh(mu)
        if act_noise > 1e-3:
            # sample gaussian noises
            noises = act_noise * torch.rand_like(actions)
            # noises = torch.clamp(noises, -1 - actions, 1 - actions)
            if clip_noise is not None:
                noises = torch.clamp(noises, -clip_noise, clip_noise)
            # add noise to actions
            actions = actions + noises
        # re-scale and re-location of actions
        return self.action_scale * actions + self.action_bias
    
    # select action strategy methods --------------------------------------
    @torch.no_grad()
    def sample(self, input_dict: Dict[str, Any], *, act_noise: float = 0.1) -> torch.Tensor:
        """ Sample actions from the policy. """
        return self(input_dict, act_noise=act_noise).clamp(self.action_low, self.action_high)
    
    @torch.no_grad()
    def select_best(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        """ Select the best action based on the policy. """
        return self(input_dict, act_noise=0.0)
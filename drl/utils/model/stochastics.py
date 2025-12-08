from typing import Callable, Tuple, Dict, Any

import math
from gymnasium import spaces

import torch
from torch import nn
from torch.nn.functional import log_softmax, softplus


##############################################################################################
################################# BASE STOCHASTIC POLICY #####################################
##############################################################################################
class StochasticPolicyBase(nn.Module):
    """ Base class for stochastic policies. """
    def __init__(
        self, 
        make_policy_fn: Callable[[Any], nn.Module], 
        observation_space: spaces.Space,
        action_space: spaces.Space,
    ):
        super().__init__()
        assert isinstance(observation_space, spaces.Space), TypeError(f'observation_space must be {type(spaces.Space)}!!!')
        assert isinstance(action_space, spaces.Space), TypeError(f'action_space must be {type(spaces.Space)}!!!')
        self.policy_net = make_policy_fn(
            observation_space=observation_space, 
            action_space=action_space
        )
        assert isinstance(self.policy_net, nn.Module), TypeError(f'Invalid policy_net type. Policy_net type must be {nn.Module}!!!')

    # compute distribution parameters method ------------------------------
    def forward(self, *args, **kwargs):
        """ Forward method to be overridden. """
        raise NotImplementedError

    # select action strategy methods --------------------------------------
    @torch.no_grad()
    def sample(self, *args, **kwargs):
        """ Method to sample random actions according to a certain distribution. """
        raise NotImplementedError

    @torch.no_grad()    
    def select_best(self, *args, **kwargs):
        """ Method to get deterministic actions. """
        raise NotImplementedError

    # distribution utility methods ----------------------------------------
    @staticmethod
    def rsample(*args, **kwargs):
        """ Method to reparameterize random sample. """
        raise NotImplementedError

    @staticmethod
    def log_prob(*args, **kwargs):
        """ Method to calculate log probability of actions. """
        raise NotImplementedError
    
    @staticmethod
    def entropy(*args, **kwargs):
        """ Method to compute entropy. """
        raise NotImplementedError
    
    @staticmethod
    def kl_divergence(*args, **kwargs):
        """ Method to compute KL divergence. """
        raise NotImplementedError


##############################################################################################
################################# MULTINOMIAL DISTRIBUTION ###################################
##############################################################################################
class MultinomialPolicy(StochasticPolicyBase):
    """ Multinomial Policy for Discrete Action Space. """
    def __init__(
        self, 
        make_policy_fn: Callable[[Any], nn.Module], 
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        eps: float = 1e-8
    ):
        super().__init__(
            make_policy_fn=make_policy_fn,
            observation_space=observation_space,
            action_space=action_space
        )
        self.register_buffer('start_action', torch.as_tensor(data=action_space.start, dtype=torch.long))
        self.eps = eps

    # compute distribution parameters method ------------------------------
    def forward(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        """ Forward method to compute action probabilities. """
        feats = self.policy_net(input_dict)
        log_probs = log_softmax(feats, dim=-1)
        probs = log_probs.exp().clamp(self.eps, 1 - self.eps)
        return probs, log_probs
    
    # select action strategy methods --------------------------------------
    @torch.no_grad()
    def sample(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        """ Sample actions from the policy. """
        feats = self.policy_net(input_dict)
        log_probs = log_softmax(feats, dim=-1)
        # re-start value of actions
        return self.rsample(log_probs) + self.start_action
    
    @torch.no_grad()
    def select_best(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        """ Select the best action based on the policy. """
        feats = self.policy_net(input_dict)
        return feats.argmax(dim=-1) + self.start_action

    # distribution utility methods ----------------------------------------
    @staticmethod
    def rsample(log_probs: torch.Tensor) -> torch.Tensor:
        """ Reparameterize random sample. """
        return torch.multinomial(log_probs.exp(), 1).view(-1)
    
    @staticmethod
    def log_prob(actions: torch.Tensor, log_probs: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        """ Calculate log probability of actions. """
        log_probs_a = torch.gather(log_probs, -1, actions.unsqueeze(-1))
        if not keepdim:
            log_probs_a = log_probs_a.flatten(-2, -1)
        return log_probs_a
    
    @staticmethod
    def kl_divergence(log_probs1: torch.Tensor, log_probs2: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        """ Compute KL divergence. """
        return (log_probs1.exp() * (log_probs1 - log_probs2)).sum(dim=-1, keepdim=keepdim)


##############################################################################################
################################### GAUSSIAN DISTRIBUTION ####################################
##############################################################################################
class GaussianPolicy(StochasticPolicyBase):
    """ Gaussian Policy for Continuous Action Space. """
    def __init__(
        self, 
        make_policy_fn: Callable[[Any], nn.Module], 
        observation_space: spaces.Space,
        action_space: spaces.Box,
        std_range: Tuple[float, float] = (0.0, 2.0)
    ):
        super().__init__(
            make_policy_fn=make_policy_fn,
            observation_space=observation_space,
            action_space=action_space
        )
        self.register_buffer('action_low', torch.as_tensor(data=action_space.low, dtype=torch.float32))
        self.register_buffer('action_high', torch.as_tensor(data=action_space.high, dtype=torch.float32))
        self.register_buffer('action_scale', torch.as_tensor(data=(action_space.high - action_space.low) / 2, dtype=torch.float32))
        self.register_buffer('action_bias', torch.as_tensor(data=(action_space.high + action_space.low) / 2, dtype=torch.float32))
        self.std_range = std_range

    # compute distribution parameters method ------------------------------
    def forward(self, input_dict: Dict[str, Any]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """ Forward method to compute action parameters. """
        # compute action parameters
        mu, log_std = self.policy_net(input_dict)
        mu, std = torch.tanh(mu), softplus(log_std)
        std = torch.clamp(std, *self.std_range)
        # reparameterized actions
        actions = self.rsample(mu, std)
        log_probs = self.log_prob(actions, mu, std, dim=-1, keepdim=True)
        # re-scale and re-location of actions
        actions = self.action_scale * actions + self.action_bias
        return (actions, log_probs), (mu, std)
    
    # select action strategy methods --------------------------------------
    @torch.no_grad()
    def sample(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        """ Sample actions from the policy. """
        # compute action parameters
        mu, log_std = self.policy_net(input_dict)
        mu, std = torch.tanh(mu), softplus(log_std)
        std = torch.clamp(std, *self.std_range)
        # reparameterized actions
        actions = torch.clamp(self.rsample(mu, std), -1.0, 1.0)
        # re-scale and re-location of actions
        return self.action_scale * actions + self.action_bias
    
    @torch.no_grad()
    def select_best(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        """ Select the best action based on the policy. """
        mu, _ = self.policy_net(input_dict)
        # re-scale and re-location of actions
        return self.action_scale * torch.tanh(mu) + self.action_bias
    
    # distribution utility methods ----------------------------------------
    @staticmethod
    def rsample(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """ Reparameterize random sample. """
        return mu + torch.randn_like(std) * std
    
    @staticmethod
    def log_prob(actions: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        """ Calculate log probability of actions. """
        z = (actions - mu) / std
        log_probs = -0.5 * (z.pow(2) + 2 * std.log() + math.log(2 * math.pi))
        return torch.sum(log_probs, dim=-1, keepdim=keepdim)
    
    @staticmethod
    def kl_divergence(mu1: torch.Tensor, std1: torch.Tensor, mu2: torch.Tensor, std2: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        """ Compute KL divergence. """
        return torch.sum(torch.log(std2 / std1) + (std1.pow(2) + (mu1 - mu2).pow(2)) / (2 * std2.pow(2)) - 0.5, dim=-1, keepdim=keepdim)


class TanhNormalPolicy(StochasticPolicyBase):
    """ Tanh Normal Policy for Continuous Action Space. """
    def __init__(
        self, 
        make_policy_fn: Callable[[Any], nn.Module], 
        observation_space: spaces.Space,
        action_space: spaces.Box,
        log_std_range: Tuple[float, float] = (-6.0, 2.0)
    ):
        super().__init__(
            make_policy_fn=make_policy_fn,
            observation_space=observation_space,
            action_space=action_space
        )
        self.register_buffer('action_low', torch.as_tensor(data=action_space.low, dtype=torch.float32))
        self.register_buffer('action_high', torch.as_tensor(data=action_space.high, dtype=torch.float32))
        self.register_buffer('action_scale', torch.as_tensor(data=(action_space.high - action_space.low) / 2, dtype=torch.float32))
        self.register_buffer('action_bias', torch.as_tensor(data=(action_space.high + action_space.low) / 2, dtype=torch.float32))
        self.log_std_range = log_std_range


    # compute distribution parameters method ------------------------------
    def forward(self, input_dict: Dict[str, Any]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """ Forward method to compute action parameters. """
        # compute action parameters
        mu, log_std = self.policy_net(input_dict)
        std = torch.exp(torch.clamp(log_std, *self.log_std_range))
        # reparameterized actions
        u = self.rsample(mu, std)
        actions = self.action_scale * torch.tanh(u) + self.action_bias
        log_probs = self.log_prob(u, mu, std, keepdim=True)
        return (actions, log_probs), (mu, std)
    
    def log_prob_action(self, actions: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # re-center and scale of actions
        actions = (actions - self.action_bias) / self.action_scale
        # convert to raw actions and compute log densities
        u = torch.arctanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
        return self.log_prob(u, mu, std, keepdim=True)
    
    # select action strategy methods --------------------------------------
    @torch.no_grad()
    def sample(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        """ Sample actions from the policy. """
        # compute action parameters
        mu, log_std = self.policy_net(input_dict)
        std = torch.exp(torch.clamp(log_std, *self.log_std_range))
        # reparameterized actions
        actions = torch.tanh(self.rsample(mu, std))
        # re-scale and re-location of actions
        return self.action_scale * actions + self.action_bias
    
    @torch.no_grad()
    def select_best(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        """ Select the best action based on the policy. """
        mu, _ = self.policy_net(input_dict)
        # re-scale and re-location of actions
        return self.action_scale * torch.tanh(mu) + self.action_bias
    
    # distribution utility methods ----------------------------------------
    @staticmethod
    def rsample(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """ Reparameterize random sample. """
        return mu + torch.randn_like(std) * std
    
    @staticmethod
    def log_prob(u: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        """ Calculate log probability of actions. """
        z = (u - mu) / (std + 1e-8)
        log_probs = torch.sum(
            -0.5 * (z.pow(2) + 2 * std.log() + math.log(2 * math.pi)) - 2 * (math.log(2) - u - softplus(-2 * u)), dim=-1, keepdim=keepdim
        )
        return torch.sum(log_probs, dim=-1, keepdim=keepdim)
    
    @staticmethod
    def kl_divergence(mu1: torch.Tensor, std1: torch.Tensor, mu2: torch.Tensor, std2: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        """ Compute KL divergence. """
        return torch.sum(torch.log(std2 / std1) + (std1.pow(2) + (mu1 - mu2).pow(2)) / (2 * std2.pow(2)) - 0.5, dim=-1, keepdim=keepdim)
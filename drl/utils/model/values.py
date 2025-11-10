from typing import Callable, Iterable, Sequence, Union, Optional, Dict, Any

import random
from gymnasium import spaces

import torch
from torch import nn

from drl.utils.general import groupby_structure


##############################################################################################
####################################### VALUE MODELS #########################################
##############################################################################################
class DeepValueFunc(nn.Module):
    """ Deep Value Function approximator for RL (Q, V, R). """
    def __init__(
        self, 
        value_type: str, 
        make_value_fn: Callable, 
        observation_space: spaces.Space,
        action_space: spaces.Space,
        use_target: bool=False
    ):
        super().__init__()
        assert value_type in ['q', 'v', 'r'], ValueError(f'Invalid value_type ({value_type}). It must be either "q", "v", or "r"!!!')
        assert isinstance(observation_space, spaces.Space), TypeError(f'observation_space must be {type(spaces.Space)}!!!')
        assert isinstance(action_space, spaces.Space), TypeError(f'action_space must be {type(spaces.Space)}!!!')
        self.value_type = value_type
        self.use_target = use_target

        # initialize neural network
        setattr(self, value_type, make_value_fn(observation_space=observation_space, action_space=action_space))
        assert isinstance(self.value_net, nn.Module), TypeError(f'{value_type} type must be {type(nn.Module)}!!!')

        # initialize target neural network
        if self.use_target:
            setattr(self, f'{value_type}_target', make_value_fn(observation_space=observation_space, action_space=action_space))
            self.target_value_net.load_state_dict(self.value_net.state_dict())
            self.target_value_net.requires_grad_(False)

    # neural networks ---------------------------------------------------------
    @property
    def value_net(self) -> nn.Module:
        return getattr(self, self.value_type)
    
    @property
    def target_value_net(self) -> Union[nn.Module, None]:
        if self.use_target:
            return getattr(self, f'{self.value_type}_target')
        return None

    # call methods -------------------------------------------------------------
    def forward(self, input_dict: Dict[str, Any], return_target: bool = False) -> torch.Tensor:
        if return_target:
            assert self.use_target, NotImplementedError('Model does not have target network')
            return self.target_value_net(input_dict)
        else:
            return self.value_net(input_dict)
        
    # utility methods ---------------------------------------------------------
    def parameters(self) -> Iterable[nn.Parameter]:
        """ Return all parameters of the value function. """
        if self.use_target:
            return {
                'source': self.value_net.parameters(),
                'target': self.target_value_net.parameters()
            }
        return self.value_net.parameters()


#####################################################################################
####################### ENSEMBLE DOUBLE DEEP Q NETWORK ##############################
#####################################################################################
class EnsembleDeepValueFunc(nn.Module):
    """ Ensemble Deep Value Function approximator for RL (Q, V). """
    def __init__(
        self, 
        value_type: str, 
        make_value_fn: Callable, 
        observation_space: spaces.Space,
        action_space: spaces.Space,
        num_ensembles: int=2, 
        use_target: bool=False
    ):
        super().__init__()
        assert num_ensembles >= 2, ValueError(f'Invalid number_ensembles ({num_ensembles}). It must be larger than or equal to 2!!!')
        self.num_ensembles = num_ensembles
        self.use_target = use_target
        self.value_nets = nn.ModuleList([
            DeepValueFunc(value_type=value_type, 
                          make_value_fn=make_value_fn, 
                          observation_space=observation_space,
                          action_space=action_space,
                          use_target=use_target)
            for _ in range(self.num_ensembles)
        ])
    
    # property methods ----------------------------------------------------------
    def __len__(self) -> int:
        return self.num_ensembles
    
    def __getitem__(self, index: int) -> nn.Module:
        return self.value_nets[index]
    
    # call methods -------------------------------------------------------------
    def forward(self, input_dict: Dict[str, Any], num_subsets: Optional[int]=None, return_target: bool=False) -> torch.Tensor:
        """ Compute the value function for a batch of inputs. """
        if num_subsets is not None:
            assert num_subsets < self.num_ensembles, ValueError(f'Number of subsets ({num_subsets}) must be less than number of ensembles ({self.num_ensembles})!!!')
            assert num_subsets >= 2, ValueError(f'number of subsets ({num_subsets}) must be larger than or equal to 2!!!')
            value_idxs = random.sample(range(self.num_ensembles), k=num_subsets)
        else:
            value_idxs = range(self.num_ensembles)
        return groupby_structure([self[i](input_dict, return_target=return_target) for i in value_idxs], torch.stack)

    # utility methods ---------------------------------------------------------    
    def parameters(self) -> Sequence[Iterable[nn.Parameter]]:
        """ Return all parameters of the value functions. """
        return [self[i].parameters() for i in range(self.num_ensembles)]
    
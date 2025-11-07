from typing import Union, Optional, Dict, Any, TypeAlias

import torch
from torch.optim.optimizer import Optimizer as Optimizer_

from drl.utils.general import Params

class Optimizer(Optimizer_):
    def __init__(
        self, 
        params: Params, 
        defaults: Dict[str, Any] = {}, 
        max_norm: Optional[float] = None, 
        accumulation: int = 1
    ):
        if max_norm is not None:
            assert max_norm > 0, ValueError(f'Invalid clip gradient norm: {max_norm}')
        assert accumulation > 0, ValueError(f'Invalid accumulation step: {accumulation}')
        
        # get source parameters
        if not isinstance(params, (list, tuple)):
            params = [params]
        src_params = []
        for p in params:
            if isinstance(p, dict):
                p = p['source']
            src_params.append({'params': p})
        # initialize optimizer
        Optimizer_.__init__(self=self, params=src_params, defaults=defaults)
        # set target parameters to optimizer
        for group, p in enumerate(params):
            if isinstance(p, dict):
                if 'target' in p:
                    self.param_groups[group]['target_params'] = list(p['target'])
        # set clip gradient norm and accumulation parameters
        self.defaults['max_norm'] = max_norm
        self.defaults['accumulation'] = accumulation
        self.defaults['step'] = 0

    # properties ------------------------------------------------------
    @property
    def source_params(self):
        _src_params = []
        for param_group in self.param_groups:
            _src_params += param_group['params']
        return _src_params
    
    @property
    def target_params(self):
        _trg_params = []
        for param_group in self.param_groups:
            if 'target_params' in param_group:
                _trg_params += param_group['target_params']
        return _trg_params
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += f'Parameter Group {i}\n'
            for key in sorted(group.keys()):
                if 'params' not in key:
                    format_string += f'    {key}: {group[key]}\n'
        format_string += ')'
        return format_string
    
    # update methos ---------------------------------------------------
    def backward(
        self, 
        loss: torch.Tensor, 
        *, 
        retain_graph: bool = False, 
        create_graph: bool = False, 
        update_target: Optional[bool] = True
    ) -> float:
        """ Performs a single optimization step. """
        # run backward to compute gradients
        loss.backward(retain_graph=retain_graph, create_graph=create_graph, inputs=self.source_params)
        self.defaults['step'] += 1
        # accumulate, clip gradients and update parameters
        if (self.defaults['step'] % self.defaults['accumulation']) == 0:
            # clip gradients
            if self.defaults['max_norm']:
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.source_params, 
                    max_norm=self.defaults['max_norm']
                )

            # update source parameters
            self.step()
            self.zero_grad()

            # update target parameters with soft update
            if update_target:
                for param_group in self.param_groups:
                    if 'target_params' in param_group:
                        polyak = param_group['polyak']
                        for src_param, trg_param in zip(param_group['params'], param_group['target_params']):
                            trg_param.data.copy_((1 - polyak) * trg_param.data + polyak * src_param.data)
        return loss.item()
    
    # utility methods ---------------------------------------------------------        
    def reset(self):
        """ Resets the optimizer state. """
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """ Returns the state of the optimizer as a dict."""
        state_dict_ = super().state_dict()
        for param_group in state_dict_['param_groups']:
            if 'target_params' in param_group:
                param_group.pop('target_params')
        return state_dict_
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """ Loads the optimizer state."""
        trg_params = [param_group.pop('target_params', None) for param_group in self.param_groups]
        super().load_state_dict(state_dict)
        for i, trg_param in enumerate(trg_params):
            if trg_param is not None:
                self.param_groups[i]['target_params'] = trg_param
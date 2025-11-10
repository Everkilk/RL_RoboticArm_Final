from typing import Union, Tuple, Any, Dict, Callable, Optional, TypeAlias

import torch
import numpy as np
from gymnasium import spaces

from drl.utils.optim.base import Optimizer
from drl.utils.model.coefficients import Coefficients

from drl.utils.general import LOGGER, map_structure, Device


class Agent:
    """ Agent Base for defining RL algorithms """
    def __init__(
        self, 
        observation_space: spaces.Space, 
        action_space: spaces.Space, 
        target_policy_path: str,
        *,
        models: Dict[str, torch.nn.Module],
        coefs: Optional[Dict[str, torch.nn.Module]],
        make_optimizer_fn: Callable[[torch.nn.Module, str], Optimizer],
        device: Device = None
    ):
        # model properties
        self._model_names: Tuple[str, ...] = []
        self._coef_names: Tuple[str, ...] = []
        self._optim_names: Tuple[str, ...] = []
        self._param_names: Tuple[str, ...] = []
        self._policy_path: Optional[str] = None
        self.device: Device = None

        # check spaces
        assert isinstance(observation_space, spaces.Space), TypeError('Invalid observation space')
        assert isinstance(action_space, spaces.Space), TypeError('Invalid action space')
        self.observation_space = observation_space
        self.action_space = action_space

        # register models
        assert isinstance(models, dict), TypeError(f'Invalid models type ({type(models)}). Must be a dict.')
        for model_name, model in models.items():
            self.register_model(model_name=model_name, model=model)
            self.register_optimizer(
                optim_name=f'{model_name}_optim',
                optim=make_optimizer_fn(params=model.parameters(), model_name=model_name)
            )

        # register coefficients
        if coefs is not None:
            assert isinstance(coefs, dict), TypeError(f'Invalid coefficients type ({type(coefs)}). Must be a dict.')
            for coef_name, coef in coefs.items():
                self.register_coef(coef_name=coef_name, coef=coef)
            self.register_optimizer(
                optim_name='coef_optim',
                optim=make_optimizer_fn(
                    params=[getattr(self, coef_name).parameters() for coef_name in self._coef_names],
                    model_name='coef'
                )   
            )

        # set target learning policy path
        self.set_policy_path(target_policy_path)

        # set device
        self.to(device)

    # forward methods ---------------------------------------------------------------
    @torch.inference_mode()
    def __call__(self, input_dict: Dict[str, Any], *, deterministic: bool = False, **kwargs) -> Any:
        """ Call method for computing actions. """
        assert isinstance(input_dict, dict), TypeError('Input data must be a dictionary')
        input_dict = map_structure(self.format_data, input_dict)
        return self.forward(input_dict=input_dict, deterministic=deterministic, **kwargs)

    def forward(self, *args, **kwargs):
        """ Forward method for computing actions. """
        raise NotImplementedError
    
    # backward methods -------------------------------------------------------------
    def update(
        self, 
        data: Dict[str, Any], 
        **kwargs,
    ) -> Union[Tuple[Dict[str, float]], Tuple[Dict[str, float], torch.Tensor]]:
        """ Update method for training the agent with given data. """
        assert isinstance(data, dict), TypeError(f'Input data must be a dictionary ({type(data)}).')
        # check device
        data = map_structure(self.format_data, data)
        # update models
        return self.backward(data, **kwargs)
    
    def backward(self, *args, **kwargs):
        """ Backward method for training the agent with given data. """
        raise NotImplementedError
    
    # estimate values / evaluate actions ------------------------------------------
    @torch.inference_mode()
    def evaluate_action(self, actions: Any, input_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """ Evaluate actions for given input data. """
        raise NotImplementedError

    @torch.inference_mode()
    def compute_action_value(self, input_dict: Dict[str, Any], actions: Any) -> torch.Tensor:
        """ Compute action value for given actions. """
        raise NotImplementedError
    
    @torch.inference_mode()
    def compute_state_value(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        """ Compute state value for given input data. """
        raise NotImplementedError
    
    # register methods -------------------------------------------------------------
    def register_model(self, model_name: str, model: torch.nn.Module):
        assert isinstance(model, torch.nn.Module), TypeError(f'Invalid model type ({type(model)}).')
        if model_name not in self._model_names:
            self._model_names.append(model_name)
        setattr(self, model_name, model.to(self.device))
        return self
    
    def register_coef(self, coef_name: str, coef: torch.nn.Module):
        assert isinstance(coef, Coefficients), TypeError(f'Invalid coef model type ({type(coef)}).')
        if coef_name not in self._coef_names:
            self._coef_names.append(coef_name)
        setattr(self, coef_name, coef.to(self.device))
        return self

    def register_optimizer(self, optim_name: str, optim: Optimizer):
        assert isinstance(optim, Optimizer), TypeError(f'Invalid optimizer type ({type(optim)}).')
        if optim_name not in self._optim_names:
            self._optim_names.append(optim_name)
        setattr(self, optim_name, optim)
        return self
    
    def register_param(self, param_name: str, param: Union[str, bool, int, float]):
        assert isinstance(param, (str, bool, int, float)), TypeError(f'Invalid param type ({type(param)}). Must be str, bool, int or float.')
        if param_name not in self._param_names:
            self._param_names.append(param_name)
        setattr(self, f'_{param_name}', param)
        return self

    def set_policy_path(self, policy_path: str):
        assert isinstance(eval(f'self.{policy_path}'), torch.nn.Module), TypeError(f'Invalid the policy path ({policy_path}).')
        self._policy_path = policy_path

    # utility methods --------------------------------------------------------------
    def format_data(self, data: Union[torch.Tensor, np.ndarray, np.number]) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            if data.dtype == torch.float64:
                data = data.to(torch.float32)
        elif isinstance(data, (np.ndarray)):
            if data.dtype == np.float64:
                data = data.astype(np.float32)
            data = torch.from_numpy(data)
        elif isinstance(data, (int, float, bool, np.number)):
            data = torch.as_tensor(data=data)
            if data.dtype == torch.float64:
                data = data.to(torch.float32)
        else:
            raise TypeError(f'Only support NumpyArray and Tensor type!!! Not support {type(data)}')
        if data.device != self.device:
            data = data.to(self.device)
        return data
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            'models': {
                model_name: getattr(self, model_name).state_dict()
                for model_name in self._model_names
            },
            'coefs': {
                coef_name: getattr(self, coef_name).state_dict()
                for coef_name in self._coef_names
            },
            'optimizers': {
                optim_name: getattr(self, optim_name).state_dict()
                for optim_name in self._optim_names
            },
            'params': {
                param_name: getattr(self, f'_{param_name}')
                for param_name in self._param_names
            },
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        LOGGER.info('Loading agent state dict ...')
        
        LOGGER.info('- Loading models:')
        for model_name in self._model_names:
            assert model_name in state_dict['models'], KeyError(f'{model_name} is not in the state_dict.')
            getattr(self, model_name).load_state_dict(state_dict['models'][model_name])
            LOGGER.info(f'\t+ Loaded state dict for {model_name}.')

        LOGGER.info('- Loading coefficients:')
        for coef_name in self._coef_names:
            assert coef_name in state_dict['coefs'], KeyError(f'{coef_name} is not in the state_dict.')
            getattr(self, coef_name).load_state_dict(state_dict['coefs'][coef_name])
            LOGGER.info(f'\t+ Loaded state dict for {coef_name}.')
        
        LOGGER.info('- Loading optimizers:')
        for optim_name in self._optim_names:
            assert optim_name in state_dict['optimizers'], KeyError(f'{optim_name} is not in the state_dict.')
            getattr(self, optim_name).load_state_dict(state_dict['optimizers'][optim_name])
            LOGGER.info(f'\t+ Loaded state dict for {optim_name}.')

        LOGGER.info('- Loading parameters:')
        for param_name in self._param_names:
            assert param_name in state_dict['params'], KeyError(f'{param_name} is not in the state_dict.')
            param_val = state_dict['params'][param_name]
            setattr(self, f'_{param_name}', param_val)
            LOGGER.info(f'\t+ Loaded parameter {param_name}: {param_val}.')
        
        return self

    def load_policy_state_dict(self, state_dict: Dict[str, Any]):
        assert self._policy_path is not None, NotImplementedError(f'policy_path is not set.')
        policy = eval(f'self.{self._policy_path}')
        if 'policy' in state_dict:
            policy.load_state_dict(state_dict['policy'])
        else:
            policy.load_state_dict(state_dict)
        LOGGER.info(f'Loaded state dict for policy at {self._policy_path}.')
        return self
    
    def policy_state_dict(self) -> Dict[str, Any]:
        assert self._policy_path is not None, NotImplementedError(f'policy_path is not set.')
        return eval(f'self.{self._policy_path}').state_dict()
    
    def to(self, device: Device):
        if device is None:
            device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device), TypeError(f"device must be a string or torch.device, not {type(device)}")
        for model_name in self._model_names:
            getattr(self, model_name).to(device)
        for coef_name in self._coef_names:
            getattr(self, coef_name).to(device)
        self.device = device
        return self
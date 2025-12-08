from typing import Tuple, Union, Optional, Dict, Any

import torch
import numpy as np

from drl.utils.general import LOGGER, map_structure, Device, Buffer


class MemoryBuffer:
    def __init__(
        self,
        max_length: int = 100000,
        use_priority: bool = False,
        device: Device = None
    ):
        self._buffers: Dict[str, Any] = {}
        self._pointer: int = 0
        self._buffer_length: int = 0
        self._is_built: bool = False
        self._priorities: Optional[torch.Tensor] = None
        self._param_names: Tuple[str, ...] = []
        assert max_length > 0, ValueError(f"max_length ({max_length}) must be greater than 0.")
        self.max_length = max_length
        self.use_priority = use_priority
        self.to(device)

    # override methods: store, sample, update priorities  ----------------------------------------------------
    def store(self, *args, **kwargs):
        """ Store batch transitions into the memory. """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """ Sample batch transitions from the memory. """
        raise NotImplementedError
    
    def get(self, *args, **kwargs):
        """ Get batch transitions from the memory. """
        raise NotImplementedError
    
    def update_priorities(self, *args, **kwargs):
        """ Update priorities for selected transitions. """
        raise NotImplementedError

    # utility methods ----------------------------------------------------------------------------------------
    def __len__(self):
        return self._buffer_length
    
    def __getitem__(self, buffer_name: str) -> Buffer:
        return map_structure(lambda x: x[:self._buffer_length], self._buffers[buffer_name])

    def _format(self, buffer: Union[torch.Tensor, np.ndarray, np.number]) -> torch.Tensor:
        if isinstance(buffer, torch.Tensor):
            if buffer.dtype == torch.float64:
                buffer = buffer.to(torch.float32)
        elif isinstance(buffer, (np.ndarray)):
            if buffer.dtype == np.float64:
                buffer = buffer.astype(np.float32)
            buffer = torch.from_numpy(buffer)
        elif isinstance(buffer, (int, float, bool, np.number)):
            buffer = torch.as_tensor(data=buffer)
            if buffer.dtype == torch.float64:
                buffer = buffer.to(torch.float32)
        else:
            raise TypeError(f'Only support NumpyArray and Tensor type!!! Not support {type(buffer)}')
        if buffer.device != self.device:
            buffer = buffer.to(self.device)
        return buffer

    def _build(self, example: Buffer):  
        # format example to the standard type
        example = map_structure(self._format, example)
        builder = lambda x: torch.tile(input=torch.zeros_like(x), dims=(self.max_length, *(x.ndim * [1])))
        # run build buffers
        for name, example_buffer in example.items():
            self._buffers[name] = map_structure(builder, example_buffer)
        if self.use_priority:
            self._priorities = builder(torch.zeros([], dtype=torch.float32, device=self.device))
            self._max_priority = 0.0
        return self

    def state_dict(self) -> Dict[str, Any]:
        state_dict_ = {
            'max_length': self.max_length,
            'buffers': self._buffers,
            'pointer': self._pointer,
            'buffer_length': self._buffer_length,
            'params': {name: self.get_param(name) for name in self._param_names}
        }
        if self.use_priority:
            state_dict_['priorities'] = self._priorities
        return state_dict_
    
    def set_param(self, param_name: str, param_value: Union[str, bool, int, float]):
        assert isinstance(param_value, (str, bool, int, float)), TypeError(f'Invalid type of param ({type(param_value)}).')
        if param_name not in self._param_names:
            self._param_names.append(param_name)
        setattr(self, param_name, param_value)

    def get_param(self, param_name: str) -> Union[str, bool, int, float]:
        assert param_name in self._param_names, AttributeError(f'The memory does not have {param_name} attribute.')
        return getattr(self, param_name)
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        LOGGER.info('Loading memory state dict ...')
        self.max_length = state_dict['max_length']
        self._buffers = map_structure(self._format, state_dict['buffers'])
        self._pointer = state_dict['pointer']
        self._buffer_length = state_dict['buffer_length']
        self._is_built = (self._buffer_length > 0)
        for param_name, param_value in state_dict['params'].items():
            self.set_param(param_name, param_value)
        if 'priorities' in state_dict:
            self.use_priority = True
            self._priorities = state_dict['priorities'].to(self.device)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        # show the memory information after loading state_dict
        LOGGER.info(f'- Max length: {self.max_length}')
        LOGGER.info(f'- Use priority: {self.use_priority}')
        LOGGER.info(f'- Pointer: {self._pointer}')
        LOGGER.info(f'- Buffer:')
        for buffer_name, buffer_size in map_structure(lambda x: tuple(x[:self._buffer_length].size()), self._buffers).items():
            LOGGER.info(f'\t+ {buffer_name} size: {buffer_size}')
        if len(self._param_names):
            LOGGER.info(f'- Memory Params:')
            for param_name in self._param_names:
                LOGGER.info(f'\t+ {param_name}: {self.get_param(param_name)}')
        return self
    
    def to(self, device: Device):
        if device is None:
            device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device), TypeError(f"device must be a string or torch.device, not {type(device)}")
        self.device = device    
        for name, buffer in self._buffers.items():
            self._buffers[name] = map_structure(lambda x: x.to(device), buffer)
        return self
    

    @classmethod
    def load_from_file(cls, file_path: str, **kwargs):
        """ Load memory from file. """
        LOGGER.info(f'Loading memory from {file_path} ...')
        state_dict = torch.load(file_path, weights_only=False, map_location='cpu')
        memory = cls(max_length=state_dict['max_length'], **kwargs)
        return memory.load_state_dict(state_dict)
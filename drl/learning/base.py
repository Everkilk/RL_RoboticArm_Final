from typing import Dict, Tuple, Union, Optional, Callable, TypeAlias

import yaml
import torch
import logging
from pathlib import Path

from drl.agent.base import Agent
from drl.memory.base import MemoryBuffer
from drl.utils.env_utils import VectorizedEnv, IsaacVecEnv
from drl.utils.general import LOGGER, increase_exp

AgentT: TypeAlias = Agent
MemoryBufferT: TypeAlias = MemoryBuffer
VectorizedEnvT: TypeAlias = Union[VectorizedEnv, IsaacVecEnv]


class RLFrameWork:
    """
        Base class for Reinforcement Learning Frameworks.
        It provides the basic structure for RL algorithms, including agent, memory, and environment.
    """
    def __init__(
        self, 
        envs: VectorizedEnvT,
        agent: AgentT, 
        memory: MemoryBufferT, 
        *,
        compute_metrics: Optional[Callable] = None
    ):
        assert isinstance(envs, (VectorizedEnv, IsaacVecEnv)), TypeError(f'Invalid envs type ({type(envs)})')
        assert isinstance(agent, Agent), TypeError(f'Invalid agent type ({type(agent)}).')
        assert isinstance(memory, MemoryBuffer), TypeError(f'Invalid memory type ({type(memory)}).')
        self.envs = envs
        self.agent = agent
        self.memory = memory
        # learning parameters --------------------
        if compute_metrics is None:
            compute_metrics = lambda x: x['eps_reward']
        assert callable(compute_metrics), TypeError(f'Invalid compute_metrics type ({type(compute_metrics)}). It is not a function.')
        self.compute_metrics = compute_metrics

    # exploration part -----------------------------------------
    def select_actions(self, *args, **kwargs):
        raise NotImplementedError
    
    # update part ----------------------------------------------    
    def train(self, *args, **kwargs):
        raise NotImplementedError
    
    # running flow ---------------------------------------------
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError
    
    def run(self, *args, **kwargs):
        raise NotImplementedError
    
    # utility methods ------------------------------------------------------------------------
    def save_policy(self, f: Union[str, Path], best_eval: float, global_step: int, save_every_best: bool = False):
        f = Path(f)
        assert f.exists() and f.is_dir(), EOFError(f'Invalid f path ({f.as_posix()}).')
        best_weight_name = f'best_policy_{global_step}_{round(best_eval, 5)}.pt' if save_every_best else 'best_policy.pt'
        torch.save(
            obj=self.agent.policy_state_dict(),
            f=f / best_weight_name
        )
        LOGGER.info(10 * '>' + 10 * '-' + f' BEST POLICY SAVED ' + 10 * '-' + 10 * '<')
        return self
    
    def save_ckpt(self, f: Union[str, Path], running_params: Dict[str, Union[str, bool, int, float]]):
        f = Path(f)
        assert f.exists() and f.is_dir(), EOFError(f'Invalid f path ({f.as_posix()}).')
        torch.save(obj=self.memory.state_dict(), f=f / 'memory.pt')
        torch.save(obj=self.agent.state_dict(), f=f / 'agent.pt')
        with open(f / 'running_params.yaml', mode='w') as file:
            yaml.dump(data=running_params, stream=file)
        LOGGER.info(10 * '>' + 10 * '-' + f' CHECKPOINT SAVED ' + 10 * '-' + 10 * '<')
        return self

    def load_ckpt(self, f: Union[str, Path], running_param_names: Tuple[str, ...]):
        f = Path(f)
        assert f.exists() and f.is_dir(), EOFError(f'Invalid f path ({f.as_posix()}).')
        self.memory.load_state_dict(torch.load(f=f / 'memory.pt', map_location='cpu', weights_only=False))
        self.agent.load_state_dict(torch.load(f=f / 'agent.pt', map_location='cpu', weights_only=False))
        with open(f / 'running_params.yaml', mode='r') as file:
            running_params = yaml.full_load(stream=file)
            for param_name in running_param_names:
                setattr(self, f'_{param_name}', running_params[param_name])
        LOGGER.info(10 * '>' + 10 * '-' + ' CHECKPOINT LOADED ' + 10 * '-' + 10 * '<')
        return self

    def make_exp_dir(
        self, 
        project_path: Union[str, Path] = '', 
        name: str = 'exp', 
        resume_path: Union[str, Path] = '', 
        subfols: Tuple[str, ...] = ['ckpt', 'policy', 'events']
    ) -> Path:
        # build experiment folder structure
        if resume_path != '':
            exp_dir = Path(resume_path)
            for subfol in subfols:
                assert (exp_dir / subfol).exists(), EOFError(f'Invalid resume_path ({(exp_dir / subfol).as_posix()}). \
                                                    Its parent must be organized with subfolders [ckpt, policy, events].')
        else:
            exp_dir = increase_exp(project_path, name)
            for subfol in subfols:
                (exp_dir / subfol).mkdir(parents=True, exist_ok=True)
        # setup logging file for global logger
        for handler in LOGGER.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                LOGGER.removeHandler(handler)
        fileHandler = logging.FileHandler(exp_dir / 'logs.txt', mode='a', delay=True)
        fileHandler.setFormatter(logging.Formatter('%(asctime)s : %(message)s'))
        LOGGER.addHandler(fileHandler)
        return exp_dir
from typing import Dict, Tuple, Union, Optional, Callable, TypeAlias

import gc
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
    
    def save_memory_only(self, f: Union[str, Path]):
        """Save only memory buffer - useful for periodic saves during training."""
        f = Path(f)
        assert f.exists() and f.is_dir(), EOFError(f'Invalid f path ({f.as_posix()}).')
        
        memory_path = f / 'memory.pt'
        memory_temp = f / 'memory.pt.tmp'
        memory_backup = f / 'memory.pt.bak'
        
        # Backup existing
        if memory_path.exists():
            try:
                memory_path.replace(memory_backup)
            except Exception:
                pass
        
        # Save with chunking
        try:
            self._save_memory_chunked(memory_temp)
            memory_temp.replace(memory_path)
            if memory_backup.exists():
                memory_backup.unlink()
            LOGGER.info('Memory buffer saved successfully.')
        except RuntimeError as e:
            LOGGER.error(f'Failed to save memory: {e}')
            if memory_backup.exists():
                memory_backup.replace(memory_path)
        
        torch.cuda.empty_cache()
        gc.collect()
        return self
    
    def save_ckpt(self, f: Union[str, Path], running_params: Dict[str, Union[str, bool, int, float]]):
        f = Path(f)
        assert f.exists() and f.is_dir(), EOFError(f'Invalid f path ({f.as_posix()}).')
        # Atomic writes for all checkpoint files
        memory_path = f / 'memory.pt'
        memory_temp = f / 'memory.pt.tmp'
        agent_path = f / 'agent.pt'
        agent_temp = f / 'agent.pt.tmp'
        
        # Backup existing memory checkpoint before overwriting
        memory_backup = f / 'memory.pt.bak'
        if memory_path.exists():
            try:
                memory_path.replace(memory_backup)
            except Exception:
                pass  # Backup failed but continue
        
        # Save memory with chunked approach to avoid OOM
        try:
            self._save_memory_chunked(memory_temp)
            memory_temp.replace(memory_path)
            # Remove backup after successful save
            if memory_backup.exists():
                memory_backup.unlink()
        except RuntimeError as e:
            LOGGER.error(f'Failed to save memory checkpoint: {e}')
            # Restore backup if available
            if memory_backup.exists():
                memory_backup.replace(memory_path)
                LOGGER.info('Restored previous memory checkpoint from backup.')
            else:
                LOGGER.warning('No memory checkpoint available - will start with empty buffer on resume.')
        
        torch.cuda.empty_cache()
        gc.collect()
        
        torch.save(obj=self.agent.state_dict(), f=agent_temp)
        agent_temp.replace(agent_path)
        
        # Atomic write: use temp file then rename to prevent corruption
        yaml_path = f / 'running_params.yaml'
        yaml_temp = f / 'running_params.yaml.tmp'
        with open(yaml_temp, mode='w') as file:
            yaml.dump(data=running_params, stream=file)
        yaml_temp.replace(yaml_path)  # Atomic on Windows/POSIX
        LOGGER.info(10 * '>' + 10 * '-' + f' CHECKPOINT SAVED ' + 10 * '-' + 10 * '<')
        return self
    
    def _save_memory_chunked(self, path: Path, chunk_size: int = 5000):
        """Save memory buffer in chunks to avoid OOM during CPU transfer."""
        memory_state = self.memory.state_dict()
        
        # Process each tensor with chunking for large buffers
        for key in list(memory_state.keys()):
            if torch.is_tensor(memory_state[key]):
                tensor = memory_state[key]
                # If tensor is large (> 100MB), move in chunks
                if tensor.numel() * tensor.element_size() > 100_000_000:
                    LOGGER.info(f'Chunking large tensor {key} ({tensor.numel()} elements) for CPU transfer...')
                    chunks = []
                    for i in range(0, len(tensor), chunk_size):
                        chunk = tensor[i:i+chunk_size].cpu()
                        chunks.append(chunk)
                        torch.cuda.empty_cache()
                    memory_state[key] = torch.cat(chunks, dim=0)
                    del chunks
                    gc.collect()
                else:
                    memory_state[key] = tensor.cpu()
        
        # Save to file
        torch.save(obj=memory_state, f=path)
        del memory_state
        gc.collect()

    def load_ckpt(self, f: Union[str, Path], running_param_names: Tuple[str, ...], require_memory: bool = True):
        f = Path(f)
        assert f.exists() and f.is_dir(), EOFError(f'Invalid f path ({f.as_posix()}).')
        
        # Try loading memory with corruption detection
        memory_corrupted = False
        memory_path = f / 'memory.pt'
        try:
            self.memory.load_state_dict(torch.load(f=memory_path, map_location='cpu', weights_only=False))
        except (RuntimeError, EOFError, OSError) as e:
            LOGGER.error(f'Memory checkpoint corrupted: {e}')
            memory_corrupted = True
            if require_memory:
                raise RuntimeError('Memory checkpoint corrupted and require_memory=True. Cannot resume off-policy training without replay buffer. Delete checkpoint folder and start fresh.')
        
        # Try loading agent with corruption detection
        agent_path = f / 'agent.pt'
        try:
            self.agent.load_state_dict(torch.load(f=agent_path, map_location='cpu', weights_only=False))
        except (RuntimeError, EOFError, OSError) as e:
            LOGGER.error(f'Agent checkpoint corrupted: {e}')
            if memory_corrupted:
                raise RuntimeError('Both memory.pt and agent.pt are corrupted. Cannot resume training. Delete checkpoint folder and start fresh.')
            else:
                raise RuntimeError('Agent checkpoint corrupted. Cannot resume training without valid agent weights. Delete checkpoint folder and start fresh.')
        
        yaml_path = f / 'running_params.yaml'
        try:
            # Check for null bytes before parsing
            with open(yaml_path, mode='rb') as file:
                raw_data = file.read()
                if b'\x00' in raw_data:
                    raise ValueError("Corrupted YAML: contains null bytes")
            
            with open(yaml_path, mode='r') as file:
                running_params = yaml.full_load(stream=file)
                for param_name in running_param_names:
                    setattr(self, f'_{param_name}', running_params[param_name])
        except (yaml.YAMLError, ValueError, OSError) as e:
            LOGGER.warning(f'YAML checkpoint corrupted or unreadable: {e}')
            LOGGER.warning('Skipping running_params load. Training will start from default values.')
            # Set defaults so training can continue
            for param_name in running_param_names:
                if not hasattr(self, f'_{param_name}'):
                    setattr(self, f'_{param_name}', 0 if param_name != 'best_eval' else float('-inf'))
        
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
from typing import Callable, Tuple, Optional, Dict, Any

import math
import torch
import random
from collections import defaultdict

from drl.memory.base import MemoryBuffer
from drl.utils.general import (map_structure, put_structure, flatten_structure, groupby_structure, 
                               nearest_node_value, query_knn_radius, Device, Buffer)


class RHERMemory(MemoryBuffer):
    """ 
        Relay Hindsight Experience Replay Memory Buffer stores rollouts from interactions. 
        It is typically used in Off-policy GCRL algorithm for sequential tasks. 
    """
    def __init__(
        self,
        reward_func: Callable[[Any, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]],
        num_stages: int, 
        horizon: int,
        max_length: int = 10000,
        use_priority: bool = False,
        *,
        alpha: Optional[float] = 0.7,
        eps: Optional[float] = 1e-5,
        down_ratio: Optional[float] = 0.2,
        bound_ratio: Optional[float] = 0.025,
        device: Device = None
    ):
        super().__init__(
            max_length=max_length,
            use_priority=use_priority,
            device=device
        )
        assert callable(reward_func), TypeError(f'Invalid reward function. It is not a function.')
        assert num_stages > 0, ValueError(f'Invalid num_stages ({num_stages})')
        assert horizon > 0, ValueError(f'Invalid horizon ({horizon}).')
        assert 0 <= alpha <= 1, ValueError(f'Invalid alpha ({alpha}).')
        self.reward_func = reward_func
        self.set_param('num_stages', num_stages)
        self.set_param('horizon', horizon)
        self.alpha = alpha
        self.eps = eps
        self.down_ratio = down_ratio
        self.bound_ratio = bound_ratio

    # store ------------------------------
    def store(self, batch: Dict[str, Buffer]):
        """ Store batch transitions into the memory. """
        # build memory buffers if it is not built
        if not self._is_built:
            valid_horizons, num_stages = [self.horizon, self.horizon + 1], self.num_stages
            example = map_structure(lambda x: x[0], batch)
            goals = {name: example[name] for name in ['desired_goal', 'achieved_goal']}
            assert all(flatten_structure(map_structure(lambda x: len(x) in valid_horizons, example))), \
                ValueError(f'Invalid horizon ({map_structure(lambda x: len(x), example)}). All elements must be have horizon in {valid_horizons}.')
            assert all(flatten_structure(map_structure(lambda x: x.shape[1] == num_stages, goals))), \
                ValueError(f'Invalid num_stages ({map_structure(lambda x: x.shape[1], goals)}). All goals must be have num_stages in {num_stages}.')
            self._build(example=example)
            self._is_built = True

        # put buffers into the memory
        batch = map_structure(self._format, batch)
        batch_length = nearest_node_value(lambda x: len(x), batch)
        # store transition buffers
        pointers = torch.arange(self._pointer, self._pointer + batch_length) % self.max_length
        put_structure(struct_trg=self._buffers, index=pointers, struct_src=batch)

        # update pointers and buffer length
        self._pointer = (self._pointer + batch_length) % self.max_length
        self._buffer_length = min(self._buffer_length + batch_length, self.max_length)

    # sample method ----------------------
    def _sample_per_stage(
        self,
        stage_id: int, 
        batch_size: int, 
        future_p: float = 0.8, 
        n_steps: int = 1, 
        step_decay: float = 0.7
    ) -> Dict[str, Buffer]:
        """ Sample transitions of a specific stage. """
        assert 0 <= stage_id < self.num_stages, ValueError(f'Invalid stage_id ({stage_id}). It must be in [0, {self.num_stages}) segment.')
        # sample indices --------------------------------------
        if self.use_priority:
            # sample indices with priorities
            traj_ind = torch.multinomial(input=self._priorities[:self._buffer_length], num_samples=batch_size, replacement=True)
        else:
            # sample indices with uniform distribution
            traj_ind = torch.randint(low=0, high=self._buffer_length, size=(batch_size,), device=self.device)
        
        # sample time steps for trajectories - sample time steps are not terminal states
        goal_achieveds = self._buffers['goal_achieved'][traj_ind, :, stage_id] # (batch_size, horizon)
        t_ind = torch.multinomial(input=1 - goal_achieveds, num_samples=1).flatten() # (batch_size,)

        # get buffer samples --------------------------------
        buffers = {
            'observations': map_structure(lambda x: x[traj_ind, t_ind], self._buffers['observation']),
            'actions': map_structure(lambda x: x[traj_ind, t_ind], self._buffers['action']),
            'goals': map_structure(lambda x: x[traj_ind, t_ind, stage_id], self._buffers['desired_goal']),
            'task_ids': torch.full((batch_size,), fill_value=stage_id, dtype=torch.long, device=self.device)
        }

        # hindsight relabel ---------------------------------
        future_check = (
            goal_achieveds * (torch.arange(self.horizon, device=self.device).view(1, -1) - t_ind.view(-1, 1)).gt_(0)
        )
        future_check[torch.all(future_check == 0, dim=1), -1] = 1
        max_offsets = future_check.argmax(dim=-1) - t_ind + 1
    
        if future_p > 1e-3:
            # select transitions to relabel
            num_relables = round(future_p * batch_size)
            relabel_ids = random.sample(range(batch_size), k=num_relables)
            re_traj_ind, re_t_ind = traj_ind[relabel_ids], t_ind[relabel_ids]

             # sample time offsets
            t_offsets = (max_offsets[relabel_ids] * torch.rand(num_relables, device=self.device)).long()

            # relabel goals with future goals
            relabel_goals = map_structure(lambda x: x[re_traj_ind, re_t_ind + t_offsets, stage_id], self._buffers['achieved_goal'])
            put_structure(buffers['goals'], relabel_ids, relabel_goals)

        # compute rewards and check terminal state --------
        if n_steps == 1:
            buffers['next_observations'] = map_structure(lambda x: x[traj_ind, t_ind + 1], self._buffers['observation'])
            rewards, terminals = self.reward_func(buffers['next_observations'], buffers['goals'], stage_id)
            buffers['rewards'], buffers['terminals'] = rewards.view(-1, 1), terminals.view(-1, 1)
        elif n_steps > 1:
            # collect n next transition steps
            next_step_buffers, step_masks = defaultdict(list), []
            for i in range(n_steps):
                t_ind = torch.clamp(t_ind + i, 0, self.horizon - 1)
                next_observations = map_structure(lambda x: x[traj_ind, t_ind + 1], self._buffers['observation'])
                rewards, terminals = self.reward_func(next_observations, buffers['goals'], stage_id)
                # store step buffers
                next_step_buffers['next_observations'].append(next_observations)
                next_step_buffers['rewards'].append(rewards)
                next_step_buffers['terminals'].append(terminals)
                step_masks.append((t_ind + i < self.horizon))
            # update next states, rewards, and terminals
            buffers.update({
                name: map_structure(lambda x: x.swapaxes(0, 1), groupby_structure(struct, torch.stack))
                for name, struct in next_step_buffers.items()
            })
            # update step masks and calculate step_decays
            step_end_idxs = torch.argmax(buffers['terminals'], axis=1)
            term_masks = torch.cummax(buffers['terminals'], dim=1)[0]
            term_masks[torch.arange(batch_size), step_end_idxs] = 0.0
            step_masks = (1 - term_masks) * torch.stack(step_masks).swapaxes(0, 1).float()
            buffers['step_decays'] = torch.cumprod(torch.full_like(step_masks, fill_value=step_decay) * step_masks, dim=1)
        else:
            raise NotImplementedError(f'Invalid n_step ({n_steps}). Do not support this n_step value.')
        
        # return buffer samples
        return buffers
    
    def sample(
        self, 
        batch_size: int, 
        future_p: float = 0.8,
        n_steps: int = 1, 
        step_decay: float = 0.7
    ) -> Dict[str, Buffer]:
        """ Sample batch transitions from the memory. """
        assert self._buffer_length > 0, ValueError("Buffer is empty!!!")
        assert batch_size > 0, ValueError(f'Invalid batch_size value ({batch_size}).')
        assert (batch_size % self.num_stages) == 0, ValueError(f'Invalid batch_size ({batch_size}). It must be divided by num_stages ({self.num_stages})')
        assert 0 <= future_p <= 1, ValueError(f'Invalid future_p value ({future_p}).')
        assert n_steps > 0, ValueError(f'Invalid n_steps ({n_steps}).')
        assert 0 < step_decay <=1, ValueError(f'Invalid step_decay ({step_decay}).')
        
        batch_size_per_stage = batch_size // self.num_stages
        return groupby_structure([
            self._sample_per_stage(stage_id=stage_id, 
                                   batch_size=batch_size_per_stage, 
                                   future_p=future_p, 
                                   n_steps=n_steps, 
                                   step_decay=step_decay)
            for stage_id in range(self.num_stages)
        ], func=torch.cat)
    

    # update priorities ------------------------------
    def update_priorities(self):
        """ Update priorities for selected transitions. """
        if self.use_priority:
            # get achieved goal trajectories in the memory
            X = self['achieved_goal'][:, :, -1].flatten(1) # (buffer_length, horizon * goal_dim)
            assert isinstance(X, torch.Tensor), NotImplementedError(f'Only support achieved_goal ({type(X)}) with Tensor type.')
            N, k, d = self._buffer_length, round(self._buffer_length ** 0.5), min(self._buffer_length, round(self.down_ratio * X.size(1)))
            if self._buffer_length < round(self.down_ratio * X.size(1)):
                self._priorities[:N] = self.eps
            
            # down size dimensions of achieved goal trajectories with PCA
            X = torch.matmul(X, torch.pca_lowrank(X, q=d)[-1])

            # compute densitites of trajectories with kNN density estimation
            densities = ((k - 1) / N) / (torch.pow((math.pi ** 0.5) * query_knn_radius(X, X, k=k), d) / math.gamma(d / 2 + 1))

            # compute complementary probabilities
            com_probs = 1 - densities / torch.sum(densities)
            
            # update priorities
            min_rank, max_rank = round(self.bound_ratio * N), round(max(self.bound_ratio * N, (1 - self.bound_ratio) * N))
            ranks = torch.clamp(torch.unique(com_probs, return_inverse=True)[-1] + 1, min_rank, max_rank)
            self._priorities[:N] = torch.pow(ranks + self.eps, self.alpha)
        else:
            raise NotImplementedError
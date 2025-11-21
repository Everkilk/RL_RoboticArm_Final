from typing import Callable, Tuple, Optional, Dict, Any

import math
import torch
import random
from collections import defaultdict

from drl.memory.base import MemoryBuffer
from drl.utils.general import (map_structure, put_structure, flatten_structure, groupby_structure, 
                               nearest_node_value, query_knn_radius, Device, Buffer)


class HERMemory(MemoryBuffer):
    """ 
        Hindsight Experience Replay Memory Buffer stores rollouts from interactions. 
        It is typically used in Off-policy GCRL algorithm. 
    """
    def __init__(
        self,
        reward_func: Callable[[Any, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        horizon: int,
        max_length: int = 10000,
        use_priority: bool = False,
        *,
        alpha: Optional[float] = 0.5,
        eps: Optional[float] = 1e-5,
        down_ratio: Optional[float] = 0.25,
        bound_ratio: Optional[float] = 0.025,
        device: Device = None
    ):
        super().__init__(
            max_length=max_length,
            use_priority=use_priority,
            device=device
        )
        assert callable(reward_func), TypeError(f'Invalid reward function. It is not a function.')
        assert horizon > 0, ValueError(f'Invalid horizon ({horizon}).')
        assert 0 <= alpha <= 1, ValueError(f'Invalid alpha ({alpha}).')
        self.reward_func = reward_func
        self.set_param('horizon', horizon)
        self.alpha = alpha
        self.eps = eps
        self.down_ratio = down_ratio
        self.bound_ratio = bound_ratio

    # store method ------------------------------
    def store(self, batch: Dict[str, Buffer]):
        """ Store batch transitions into the memory. """
        # build memory buffers if it is not built
        if not self._is_built:
            valid_horizons = [self.horizon, self.horizon + 1]
            example = map_structure(lambda x: x[0], batch)
            assert all(flatten_structure(map_structure(lambda x: len(x) in valid_horizons, example))), \
                ValueError(f'Invalid horizon ({map_structure(lambda x: len(x), example)}). All elements must be have horizon in {valid_horizons}.')
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

    # sample method --------------------------
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
        assert 0 <= future_p <= 1, ValueError(f'Invalid future_p value ({future_p}).')
        assert n_steps > 0, ValueError(f'Invalid n_steps ({n_steps}).')
        assert 0 < step_decay <=1, ValueError(f'Invalid step_decay ({step_decay}).')

        # sample indices and compute weights
        if self.use_priority:
            # sample indices with priorities
            traj_ind = torch.multinomial(input=self._priorities[:self._buffer_length], num_samples=batch_size, replacement=True)
        else:
            # sample indices with uniform distribution
            traj_ind = torch.randint(low=0, high=self._buffer_length, size=(batch_size,), device=self.device)
        
        # sample time steps for trajectories - sample time steps are not terminal states
        goal_achieveds = self._buffers['terminal'][traj_ind]
        t_ind = torch.multinomial(input=1 - goal_achieveds, num_samples=1).flatten()
        
        # get buffer samples
        buffers = {
            'observations': map_structure(lambda x: x[traj_ind, t_ind], self._buffers['observation']),
            'actions': map_structure(lambda x: x[traj_ind, t_ind], self._buffers['action']),
            'goals': map_structure(lambda x: x[traj_ind, t_ind], self._buffers['desired_goal'])
        }

        # hindsight relabel
        if future_p > 1e-3:
            # select transitions to relabel
            num_relables = round(future_p * batch_size)
            relabel_ids = random.sample(range(batch_size), k=num_relables)
            re_traj_ind, re_t_ind = traj_ind[relabel_ids], t_ind[relabel_ids]

            # sample time offsets
            t_offsets = ((self.horizon - re_t_ind) * torch.rand(num_relables, device=self.device)).long()

            # relabel goals with future goals
            relabel_goals = map_structure(lambda x: x[re_traj_ind, re_t_ind + t_offsets], self._buffers['achieved_goal'])
            put_structure(buffers['goals'], relabel_ids, relabel_goals)

        # compute rewards and check terminal state
        if n_steps == 1:
            buffers['next_observations'] = map_structure(lambda x: x[traj_ind, t_ind + 1], self._buffers['observation'])
            rewards, goal_achieveds = self.reward_func(buffers['next_observations'], buffers['goals'])
            terminals = (self._buffers['terminal'][traj_ind, t_ind] + goal_achieveds).gt_(0.0)
            buffers['rewards'], buffers['terminals'] = rewards.view(-1, 1), terminals.view(-1, 1)
        elif n_steps > 1:
            # collect n next transition steps
            next_step_buffers, step_masks = defaultdict(list), []
            for i in range(n_steps):
                t_ind = torch.clamp(t_ind + i, 0, self.horizon - 1)
                next_observations = map_structure(lambda x: x[traj_ind, t_ind + 1], self._buffers['observation'])
                rewards, goal_achieveds = self.reward_func(next_observations, buffers['goals'])
                terminals = (self._buffers['terminal'][traj_ind, t_ind] + goal_achieveds).gt_(0.0)
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
    
    # update priorities ------------------------------
    def update_priorities(self):
        """ Update priorities for selected transitions. """
        if self.use_priority:
            X = self['achieved_goal'].flatten(1) # (buffer_length, horizon * goal_dim)
            assert isinstance(X, torch.Tensor), NotImplementedError(f'Only support achieved_goal ({type(X)}) with Tensor type.')
            if self._buffer_length < round(self.down_ratio * X.size(1)):
                return
            N, k, d = self._buffer_length, round(self._buffer_length ** 0.5), min(self._buffer_length, round(self.down_ratio * X.size(1)))
            
            # down size dimensions of achieved goal horizon with PCA
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
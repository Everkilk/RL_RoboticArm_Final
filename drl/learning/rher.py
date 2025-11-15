from typing import Dict, Any, Union, Callable, Tuple, Optional

import sys
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from drl.learning.base import RLFrameWork, AgentT, MemoryBufferT, VectorizedEnvT
from drl.utils.general import (LOGGER, format_time, format_tabulate, map_structure, 
                               put_structure, groupby_structure, nearest_node_value, MeanMetrics)


class RHER(RLFrameWork):
    def __init__(
        self, 
        envs: VectorizedEnvT, 
        agent: AgentT, 
        memory: MemoryBufferT, 
        *,
        compute_metrics: Optional[Callable] = None
    ):
        if compute_metrics is None:
            compute_metrics = lambda x: x.get('goal_achieved', 0.0) + (x['eps_reward'] / x['eps_horizon'])
        super().__init__(envs=envs, agent=agent, memory=memory, compute_metrics=compute_metrics)
        # check valid agent
        assert hasattr(self.agent, 'compute_action_value'), AttributeError('The agent must be have compute_action_value method.')
        # check valid memory
        assert hasattr(self.memory, 'reward_func'), AttributeError('The memory must be have reward_func attribute.')
        assert hasattr(self.memory, 'num_stages'), AttributeError('The memory must be have num_stages attribute.')
        assert hasattr(self.memory, 'horizon'), AttributeError('The memory must be have horizon attribute.')
        # training params
        self._start_epoch: int = 1
        self._time_delta: float = 0.0
        self._best_eval: float = -float('inf')
        self._total_updates: int = 0
        
    # exploration part ----------------------------------------------------
    def _get_stages(self, observations: Any, mt_goals: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        reward_func, num_stages = self.memory.reward_func, self.memory.num_stages
        # check goal achieved for all stages
        goal_achieveds = torch.stack([
            reward_func(observations, map_structure(lambda x: x[:, stage_id], mt_goals), stage_id)[-1]
            for stage_id in range(num_stages)
        ], dim=-1) # (batch_size, num_stages)
        # select current stages
        stages = (goal_achieveds.fliplr().cummax(dim=1).values == 1).sum(dim=1).clamp(0, num_stages - 1)
        return stages, goal_achieveds

    def _select_action(self, observations: Any, mt_goals: Any, stages: torch.Tensor, *, r_mix: float = 0.5) -> torch.Tensor:
        assert 0 <= r_mix <= 1, ValueError(f'Invalid r_mix ({r_mix}).')
        batch_size = nearest_node_value(lambda x: len(x), mt_goals)
        num_stages, device = self.memory.num_stages, self.agent.device
        batch_ind = torch.arange(batch_size, device=device)
        # check stage of the process
        # mix the current stage with the next stage for guiding
        mix_masks = torch.rand(batch_size, device=device) < r_mix
        stages[mix_masks] = torch.clamp(stages[mix_masks] + 1, None, num_stages - 1)
        # get goals for explorating
        goals = map_structure(lambda x: x[batch_ind, stages], mt_goals)
        # select actions with current policy
        return self.agent({'observation': observations, 'goal': goals, 'task_id': stages}, deterministic=False)

    def select_actions(
        self, 
        observations: Any, 
        mt_goals: Any, 
        *, 
        r_mix: float = 0.5, 
        deterministic: bool = False
    ) -> Union[Tuple[Any, np.ndarray], Any]:
        if not deterministic:
            observations = map_structure(self.agent.format_data, observations)
            mt_goals = map_structure(self.agent.format_data, mt_goals)
            stages, goal_achieveds = self._get_stages(observations, mt_goals)
            actions = self._select_action(observations, mt_goals, stages, r_mix=r_mix)
            actions = map_structure(lambda x: x.cpu().numpy(), actions)
            goal_achieveds = goal_achieveds.cpu().numpy().astype(bool)
            return actions, goal_achieveds
        # get final goals and select actions with the policy
        goals = map_structure(lambda x: x[:, -1], mt_goals)
        stages = torch.full((len(goals),), self.memory.num_stages - 1, dtype=torch.long, device=self.agent.device)
        actions = self.agent({'observation': observations, 'goal': goals, 'task_id': stages}, deterministic=True)
        return map_structure(lambda x: x.cpu().numpy(), actions)
    
    # update part ---------------------------------------------------------
    def train(
        self, 
        num_updates: int, 
        batch_size: int, 
        future_p: float = 0.8,
        n_steps: int = 1,
        step_decay: float = 0.7,
        discounted_factor: float = 0.99, 
        clip_return: Optional[float] = None
    ) -> Dict[str, Any]:
        # update priorities for replay buffer
        if self.memory.use_priority:
            self.memory.update_priorities()
                    
        avg_evals = MeanMetrics()  
        with tqdm(range(num_updates), bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}', \
                  desc='- Training', leave=False) as pbar:
            for _ in pbar:
                # sample data from the memory
                data = self.memory.sample(batch_size=batch_size, 
                                          future_p=future_p, 
                                          n_steps=n_steps,   
                                          step_decay=step_decay)
                # run train step
                evals = self.agent.update(data=data, 
                                          discounted_factor=discounted_factor,
                                          clip_return=clip_return)
                # update metrics
                avg_evals.update(evals)

                # show metrics
                pbar.set_postfix(evals)

        return avg_evals

    
    # learning flow -------------------------------------------------------
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        self.envs.eval()

        # initialize params for running
        num_envs, horizon = self.envs.num_envs, self.memory.horizon
        eval_info = MeanMetrics()
        eps_rewards = np.zeros(num_envs, dtype='float32')
        eps_horizons = np.zeros(num_envs, dtype='int32')
        count_episodes = 0

        # start evaluation running
        observations, _ = self.envs.reset()
        while True:
            # select the deterministic actions
            actions = self.select_actions(observations['observation'], observations['desired_goal'], deterministic=True)

            # apply the selected action into the environment and compute episode rewards
            next_observations, rewards, ternimateds, truncateds, infos = self.envs.step(actions)
            eps_rewards += rewards
            eps_horizons += 1

            # forward the next state
            observations = next_observations
            env_ids = np.where(ternimateds | truncateds | (eps_horizons == horizon))[0]
            if len(env_ids):
                for env_id in env_ids:
                    # reset environment
                    put_structure(observations, env_id, self.envs.single_reset(env_id)[0])
                    # get episode information and compute evaluation metric
                    eps_info = map_structure(lambda x: x[env_id], infos)
                    eps_info = {name: value for name, value in eps_info.items() if value.dtype != np.object_}
                    eps_info.update({
                        'eps_reward': eps_rewards[env_id], 
                        'eps_horizon': eps_horizons[env_id]
                    })
                    eps_info.update({'eval_value': self.compute_metrics(eps_info)})
                    eval_info.update(eps_info)
                    # reset episode-specific counters
                    eps_rewards[env_id], eps_horizons[env_id] = 0.0, 0
                    count_episodes += 1
                    if count_episodes == num_episodes:
                        break
            
            # check evaluating done
            if count_episodes == num_episodes:
                break
        
        # return results
        return eval_info
    
    def run(
        self,         
        epochs: int = 1000,
        num_cycles: int = 100,
        num_eval_episodes: int = 20,
        r_mix: float = 0.5,
        *,
        # train params
        num_updates: int = 50,
        batch_size: int = 256,
        future_p: float = 0.8,
        n_steps: int = 1,
        step_decay: float = 0.7,
        discounted_factor: float = 0.98, 
        clip_return: Optional[float] = None,
        # experiment ckpt
        project_path: Union[str, Path]='',
        name: str='exp',
        resume_path: Union[str, Path] = '',
        save_every_best: bool = False
    ):
        num_envs = self.envs.num_envs
        assert (num_cycles % num_envs) == 0, ValueError(f'Invalid num_cycles ({num_cycles}). It must be divided by num_envs ({num_envs}).')
        t0 = time.time()

        # make experiement directory --------------------------------------------
        exp_dir = self.make_exp_dir(project_path, name, resume_path, subfols=['ckpt', 'policy', 'events'])
        if resume_path != '':
            LOGGER.info(10 * '>' + 10 * '-' + ' LOADING CHECKPOINT ' + 10 * '-' + 10 * '<')
            self.load_ckpt(f=exp_dir / 'ckpt', running_param_names=['start_epoch', 'best_eval', 'total_updates', 'time_delta'])
        LOGGER.info(f'*** Experiment directory: {exp_dir.as_posix()}')

        LOGGER.info(10 * '>' + 10 * '-' + ' START TRAINING ' + 10 * '-' + 10 * '<')
        LOGGER.info(f'- Start epoch: {self._start_epoch}')
        LOGGER.info(f'- Best evaluation: {self._best_eval}')
        LOGGER.info(f'- Total updates: {self._total_updates}')
        LOGGER.info(f'- Start time: {format_time(self._time_delta)}')

        with SummaryWriter(log_dir=exp_dir / 'events') as writer:
            for epoch in range(self._start_epoch, epochs + 1):
                # run explorating and training --------------------------------------------
                epoch_t0, train_run_info = time.time(), MeanMetrics()
                for i, cycle in enumerate(range(num_envs, num_cycles + num_envs, num_envs)):
                    cycle_t0 = time.time()
                    # generate rollouts with policy
                    batch_rollouts, run_info = self.generate_rollouts(r_mix=r_mix)
                    train_run_info.update(run_info)
                    # store generated rollouts
                    self.store_rollouts(batch_rollouts)
                    # run training
                    train_evals = self.train(
                        num_updates=num_updates, 
                        batch_size=batch_size,
                        future_p=future_p,
                        n_steps=n_steps,
                        step_decay=step_decay,
                        discounted_factor=discounted_factor,
                        clip_return=clip_return
                    )
                    self._total_updates += num_updates
                    for name, value in train_evals.items():
                        writer.add_scalar(
                            tag=f'train/{name}', scalar_value=value, 
                            global_step=i + (num_cycles // num_envs) * (epoch - 1)
                        )
                    # show results
                    LOGGER.info(format_tabulate(
                        title='RUNNING RESULTS',
                        subtitle=f'epoch: {epoch}/{epochs}, cycle: {cycle}/{num_cycles}, run_time: {round(time.time() - cycle_t0, 2)}s',
                        results=run_info,
                        tail_info=f'total_updates: {self._total_updates}, duration: {format_time(time.time() - t0 + self._time_delta)}'
                    ) + '\n')

                # show train result per epoch
                LOGGER.info(format_tabulate(
                    title=f'TRAINING RESULTS',
                    subtitle=f'epoch: {epoch}/{epochs}, run_time: {format_time(time.time() - epoch_t0)}',
                    results=train_run_info,
                    tail_info=f'total_updates: {self._total_updates}, duration: {format_time(time.time() - t0 + self._time_delta)}'
                ) + '\n')

                # run evaluating and compute evaluation metric --------------------------------------
                eval_t0 = time.time()
                eval_run_info = self.evaluate(num_eval_episodes)
                eval_value = eval_run_info['eval_value']
                # show evaluating results per epoch
                LOGGER.info(format_tabulate(
                    title='EVALUATING RESULTS',
                    subtitle=f'epoch: {epoch}/{epochs}, eval: {round(eval_value, 5)}, run_time: {format_time(time.time() - eval_t0)}',
                    results=eval_run_info,
                    tail_info=f'total_updates: {self._total_updates}, duration: {format_time(time.time() - t0 + self._time_delta)}'
                ))
                
                # update training parameters and save checkpoint
                if eval_value >= self._best_eval:
                    self._best_eval = float(eval_value)
                    # Save to experiment directory
                    self.save_policy(
                        f=exp_dir / 'policy', 
                        best_eval=self._best_eval, 
                        global_step=epoch, 
                        save_every_best=save_every_best
                    )
                    best_policy_dir = Path(project_path) /'runs'/ 'best_policy' / 'lift'
                    best_policy_dir.mkdir(parents=True, exist_ok=True)
                    self.save_policy(
                        f=best_policy_dir,
                        best_eval=self._best_eval,
                        global_step=epoch,
                        save_every_best=False
                    )
                        
                self.save_ckpt(f=exp_dir / 'ckpt', running_params={
                    'start_epoch': epoch + 1,
                    'best_eval': self._best_eval,
                    'total_updates': self._total_updates,
                    'time_delta': time.time() - t0 + self._time_delta,
                })
                for name in eval_run_info.keys():
                    writer.add_scalars(
                        main_tag=f'evaluations/{name}',
                        tag_scalar_dict={'train': train_run_info[name], 'eval': eval_run_info[name]},
                        global_step=epoch
                    )
                LOGGER.info('')

    # class private methods --------------------------------------------------------------
    def store_rollouts(self, rollouts: Dict[str, Any]):
        if 'meta' in rollouts:
            rollouts['meta'] = map_structure(lambda x: x[:, 1:], rollouts['meta'])
        rollouts['achieved_goal'] = map_structure(lambda x: x[:, 1:], rollouts['achieved_goal'])
        rollouts['desired_goal'] = map_structure(lambda x: x[:, :-1], rollouts['desired_goal'])
        rollouts['goal_achieved'] = np.float16(rollouts['goal_achieved'])
        self.memory.store(rollouts)

    def generate_rollouts(self, r_mix: Union[str, float] = 'auto') -> Tuple[Dict[str, Any], Dict[str, float]]:
        self.envs.train()

        # initialize params for running
        rollouts = defaultdict(list)
        eps_rewards = np.zeros(self.envs.num_envs, dtype='float32')
        eps_horizons = np.zeros(self.envs.num_envs, dtype='int32')
        
        # run generating
        observations = self.envs.reset()[0]
        for name, value in observations.items():
            rollouts[name].append(value)
        
        for _ in range(self.memory.horizon):
            # select actions from the policy
            actions, goal_achieveds = self.select_actions(
                observations['observation'], observations['desired_goal'], 
                r_mix=r_mix, deterministic=False
            )

            # apply the selected actions into environments and go to next observations
            observations, rewards, terminateds, _, infos = self.envs.step(actions)
            eps_rewards += rewards
            eps_horizons += 1

            # store transitions
            rollouts['action'].append(actions)
            for name, value in observations.items():
                rollouts[name].append(value)
            rollouts['goal_achieved'].append(goal_achieveds | terminateds.reshape(-1, 1))
        
        # format and return batch rollouts
        rollouts = {
            name: map_structure(lambda x: np.swapaxes(x, 0, 1), groupby_structure(rollout, func=np.stack))
            for name, rollout in rollouts.items()
        }
        infos = {name: value for name, value in infos.items() if value.dtype != np.object_}
        info = map_structure(lambda x: np.mean(x), {'eps_reward': eps_rewards, 'eps_horizon': eps_horizons, **infos})
        return rollouts, info
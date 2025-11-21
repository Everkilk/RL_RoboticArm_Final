from typing import Dict, Any, Union, Callable, Tuple, Optional

import time
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from drl.learning.base import RLFrameWork, AgentT, MemoryBufferT, VectorizedEnvT
from drl.utils.general import (LOGGER, format_time, format_tabulate, put_structure, 
                               map_structure, groupby_structure, MeanMetrics)


class HER(RLFrameWork):

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
        super().__init__(
            envs=envs,
            agent=agent, 
            memory=memory,
            compute_metrics=compute_metrics
        )
        # training params
        self._start_epoch: int = 1
        self._time_delta: float = 0.0
        self._best_eval: float = -float('inf')
        self._total_updates: int = 0

    # exploration part ----------------------------------------------------
    def select_actions(self, observations, goals, *, act_rd: float = 0.0, deterministic: bool = False):
        if (random.random() < act_rd) and (not deterministic):
            return self.envs.action_space.sample()
        return self.agent({'observation': observations, 'goal': goals}, deterministic=deterministic).cpu().numpy()

    # update part ---------------------------------------------------------
    def train(
        self, 
        num_updates: int, 
        batch_size: int, 
        future_p: float = 0.8,
        n_steps: int = 1,
        step_decay: float = 0.7,
        discounted_factor: float = 0.98, 
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
                data = self.memory.sample(
                    batch_size=batch_size, 
                    future_p=future_p, 
                    n_steps=n_steps, 
                    step_decay=step_decay,
                )
                
                # run train step
                evals = self.agent.update(
                    data=data, 
                    discounted_factor=discounted_factor,
                    clip_return=clip_return
                )

                # update metrics
                avg_evals.update(evals)

                # show metrics
                pbar.set_postfix(evals)

        return avg_evals

    
    # learning flow --------------------------------------------------------------
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
        act_rd: float = 0.05,
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

        # random exploration ----------------------------------------------------
        # run warming up block -----------------
        with SummaryWriter(log_dir=exp_dir / 'events') as writer:
            for epoch in range(self._start_epoch, epochs + 1):
                # run explorating and training --------------------------------------------
                epoch_t0, train_run_info = time.time(), MeanMetrics()
                for i, cycle in enumerate(range(num_envs, num_cycles + num_envs, num_envs)):
                    cycle_t0 = time.time()
                    # generate rollouts with policy
                    batch_rollouts, run_info = self.generate_rollouts(act_rd=act_rd)
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
                        writer.add_scalar(tag=f'train/{name}', scalar_value=value, 
                                        global_step=i + (num_cycles // num_envs) * (epoch - 1))
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
                    self.save_policy(
                        f=exp_dir / 'policy', 
                        best_eval=self._best_eval, 
                        global_step=epoch, 
                        save_every_best=save_every_best
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
        rollouts['terminal'] = np.float16(rollouts['terminal'][:, :-1])
        self.memory.store(rollouts)


    def generate_rollouts(self, act_rd: float = 0.2) -> Tuple[Dict[str, Any], Dict[str, float]]:
        self.envs.train()

        # initialize params for running
        rollouts = defaultdict(list)
        eps_rewards = np.zeros(self.envs.num_envs, dtype='float32')
        eps_horizons = np.zeros(self.envs.num_envs, dtype='int32')
        
        # run generating
        observations = self.envs.reset()[0]
        for name, value in observations.items():
            rollouts[name].append(value)
        rollouts['terminal'].append(np.zeros(self.envs.num_envs, dtype=bool))
        
        for _ in range(self.memory.horizon):
            # select actions from the policy
            actions = self.select_actions(
                observations['observation'], 
                observations['desired_goal'], 
                act_rd=act_rd,
                deterministic=False
            )
            # apply the selected actions into environments and go to next observations
            observations, rewards, terminateds, _, infos = self.envs.step(actions)
            eps_rewards += rewards
            eps_horizons += 1

            # store transitions
            rollouts['action'].append(actions)
            for name, value in observations.items():
                rollouts[name].append(value)
            rollouts['terminal'].append(terminateds.astype('float16'))
        
        # format and return batch rollouts
        rollouts = {
            name: map_structure(lambda x: np.swapaxes(x, 0, 1), groupby_structure(list_structs=rollout, func=np.stack))
            for name, rollout in rollouts.items()
        }
        infos = {name: value for name, value in infos.items() if value.dtype != np.object_}
        info = map_structure(lambda x: np.mean(x), {'eps_reward': eps_rewards, 'eps_horizon': eps_horizons, **infos})
        return rollouts, info
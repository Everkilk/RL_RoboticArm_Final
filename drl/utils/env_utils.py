from typing import Callable

import torch
import numpy as np
import gymnasium as gym
import multiprocessing as mp
from drl.utils.general import map_structure, groupby_structure


class VectorizedEnv(gym.Env):
    def __init__(
        self,
        make_env_fn: Callable[[int], gym.Env],
        num_envs: int,
        use_async: bool = False
    ):
        self.use_async = use_async
        self.num_envs = num_envs

        if not self.use_async:
            self.envs = [make_env_fn(env_id) for env_id in range(self.num_envs)]
            self.single_observation_space = self.envs[0].observation_space
            self.single_action_space = self.envs[0].action_space
        else:
            ctx = mp.get_context()
            # create pipes
            self.main_remotes, process_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])

            # create two process
            self.processes = []
            for env_rank, (main_remote, process_remote) in enumerate(zip(self.main_remotes, process_remotes)):
                process = mp.Process(target=self._worker, args=(process_remote, main_remote, make_env_fn, env_rank), daemon=True)
                process.start()
                self.processes.append(process)
                process_remote.close()
            self.main_remotes[0].send(('env_spaces', None))
            env_spaces = self.main_remotes[0].recv()
            self.single_observation_space = env_spaces['observation_space']
            self.single_action_space = env_spaces['action_space']

        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, n=self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, n=self.num_envs)

    def __len__(self):
        return self.num_envs

    # single environment manipulation ----------------------------------------------------------
    def single_reset(self, env_id: int):
        if self.use_async:
            self.main_remotes[env_id].send(('reset', None))
            return self.main_remotes[env_id].recv()
        else:
            return self.envs[env_id].reset()
    
    def single_step(self, action, env_id: int):
        if self.use_async:
            self.main_remotes[env_id].send(('step', action))
            return self.main_remotes[env_id].recv()
        else:
            return self.envs[env_id].step(action)

    # all environments manipulation -----------------------------------------------------------
    def reset(self):
        if self.use_async:
            for i in range(self.num_envs):
                self.main_remotes[i].send(('reset', None))
            return groupby_structure([self.main_remotes[i].recv() for i in range(self.num_envs)])
        else:
            return groupby_structure([self.envs[i].reset() for i in range(self.num_envs)])
        
    def step(self, actions):
        if self.use_async:
            for env_id, main_remote in enumerate(self.main_remotes):
                main_remote.send(('step', map_structure(lambda x: x[env_id], actions)))
            return groupby_structure([main_remote.recv() for main_remote in self.main_remotes])
        else:
            return groupby_structure([
                env.step(map_structure(lambda x: x[env_id], actions)) 
                for env_id, env in enumerate(self.envs)
            ])
        
    def close(self):
        if self.use_async:
            for main_remote in self.main_remotes:
                main_remote.send(('close', None))
        else:
            for env in self.envs:
                env.close()
        return self
        
    # mode management ------------------------------------------------------------------------
    def train(self):
        if self.use_async:
            for main_remote in self.main_remotes:
                main_remote.send(('mode', 'train'))
        else:
            for env in self.envs:
                if hasattr(env, 'train'):
                    env.train()
        return self

    def eval(self):
        if self.use_async:
            for main_remote in self.main_remotes:
                main_remote.send(('mode', 'eval'))
        else:
            for env in self.envs:
                if hasattr(env, 'eval'):
                    env.eval()
        return self
    
    # async wrapper method -------------------------------------------------------------------
    @staticmethod
    def _worker(process_remote, main_remote, make_env_fn, env_rank):
        main_remote.close()

        # create the environment
        env = make_env_fn(env_rank)

        # check valid environment
        for space in ['observation_space', 'action_space']:
            assert hasattr(env, space), AttributeError(f'env does not have {space}!!!')

        for method in ['reset', 'step', 'close']:
            assert hasattr(env, method), AttributeError(f'env does not have {method} method!!!')

        while True:
            try:
                # receive the command and data from main process
                cmd, data = process_remote.recv()

                if cmd == 'reset':
                    # send observation to main process
                    process_remote.send(env.reset())    
                elif cmd == 'step':
                    # run step from action received from main process
                    process_remote.send(env.step(data))
                elif cmd == 'mode':
                    if hasattr(env, data):
                        eval(f'env.{data}')()
                elif cmd == 'close':
                    env.close()
                    process_remote.close()
                    break
                elif cmd == 'env_spaces':
                    process_remote.send({
                        'observation_space': env.observation_space, 
                        'action_space': env.action_space
                    })
                else:
                    raise NotImplementedError
            except EOFError:
                break


#############################################################################################################
##################################### ISAAC BASED ENVIRONMENT ###############################################
#############################################################################################################
class IsaacVecEnv(gym.Env):
    def __init__(self, manager, cfg, num_envs):
        super().__init__()
        # get environment configure -----------------
        assert callable(cfg), TypeError
        self.cfg = cfg()
        self.cfg.scene.num_envs = num_envs
        # get environment manager -------------------
        assert callable(manager), TypeError
        self.manager = manager(self.cfg)
        
    @property
    def num_envs(self) -> int:
        return self.manager.num_envs

    @property
    def max_episode_length(self) -> int:
        return self.manager.max_episode_length
    
    @property
    def single_observation_space(self):
        return self.manager.single_observation_space
    
    @property
    def single_action_space(self):
        return self.manager.single_action_space
    
    @property
    def observation_space(self):
        return self.manager.observation_space
    
    @property
    def action_space(self):
        return self.manager.action_space
    
    def reset(self, seed=None, **kwargs):
        observations, infos = self.manager.reset(seed=seed, **kwargs)
        observations = map_structure(lambda x: x.cpu().numpy(), observations)
        infos = map_structure(lambda x: x.cpu().numpy(), infos)
        return observations, infos
    

    def step(self, actions, **kwargs):
        # convert to tensor for actions
        actions = torch.from_numpy(actions).to(self.manager.device)
        # run simulation step
        observations, rewards, terminated, truncated, infos = self.manager.step(actions, **kwargs)
        # format and return feedbacks
        observations = map_structure(lambda x: x.cpu().numpy(), observations)
        rewards = rewards.cpu().numpy()
        terminated = terminated.cpu().numpy()
        truncated = truncated.cpu().numpy()
        infos = map_structure(lambda x: x.cpu().numpy(), infos)
        return observations, rewards, terminated, truncated, infos

    def close(self):
        self.manager.close()

    def render(self, *args, **kwargs):
        # method is not supported
        raise NotImplementedError
    
    # mode management ------------------------------------------------------------------------
    def train(self):
        return self

    def eval(self):
        return self

    # HELPER METHODS --------------------------------------
    def single_reset(self, env_id: int):
        env_id = torch.LongTensor([env_id]).to(self.manager.device)
        self.manager._reset_idx(env_id)
        return map_structure(lambda x: x.cpu().numpy(), self.manager._get_observations(env_id.item())), {}

    def get_observations(self):
        if hasattr(self.manager, '_get_observations'):
            observations = self.manager._get_observations()
        else:
            observations = self.manager.observation_manager.compute()
        return map_structure(lambda x: x.cpu().numpy(), observations)



class IsaacEnv(gym.Env):
    def __init__(self, manager, cfg):
        super().__init__()
        # get environment configure -----------------
        self.cfg = cfg()
        self.cfg.scene.num_envs = 1
        # get environment manager -------------------
        self.manager = manager(self.cfg)
        # configure spaces --------------------------
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.num_observations,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.num_actions,))

    def reset(self, seed=None):
        observation, info = self.manager.reset(seed=seed)
        observation = map_structure(lambda x: x[0].cpu().numpy(), observation)
        info = map_structure(lambda x: x.item(), info)
        return observation, info
    

    def step(self, action):
        # convert to tensor for action
        action = torch.from_numpy(action[None]).to(self.manager.device)
        # run simulation step
        observation, reward, terminated, truncated, info = self.manager.step(action)
        # format and return feedback
        observation = map_structure(lambda x: x[0].cpu().numpy(), observation)
        reward = reward.item()
        terminated = terminated.item()
        truncated = truncated.item()
        info = map_structure(lambda x: x[0].cpu().numpy(), info)
        return observation, reward, terminated, truncated, info

    def close(self):
        self.manager.close()

    def render(self, *args, **kwargs):
        # method is not supported
        raise NotImplementedError
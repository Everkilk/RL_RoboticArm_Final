import math
import torch
from .common import *

# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm


###
##### EVENT PART
###
def reset_object_poses(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor
):
    # --- reset object state
    object_pose = sample_root_state_uniform(
        env=env, env_ids=env_ids, 
        pose_range={'x': (-0.2, 0.2), 'y': (-0.2, 0.2), 'yaw': (0, 2 * math.pi)},
        asset_cfg=SceneEntityCfg('cube')
    )[0]
    # set into the physics simulation
    env.scene['cube'].write_root_pose_to_sim(object_pose, env_ids=env_ids)


def clip_object_ranges(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor
):
    clip_object_xy_range(
        env=env, env_ids=env_ids,
        x_range=(0.05, 0.55), y_range=(-0.3, 0.3), z_thresh=0.05,
        asset_cfg=SceneEntityCfg('cube')
    )
    

@configclass
class EventsCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=reset_scene_to_default, mode="reset")

    reset_objects = EventTerm(
        func=reset_object_poses,
        mode='reset'
    )

    clip_object_positions = EventTerm(
        func=clip_object_ranges,
        mode='interact'
    )
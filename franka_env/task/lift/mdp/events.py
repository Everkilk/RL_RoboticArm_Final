import math
import torch
import numpy as np
from .common import *

# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm


###
##### EVENT PART
###

# Object names for random selection
OBJECT_NAMES = ['object_cube', 'object_mustard']  # 'object_drill' commented out in env_cfg

def reset_object_poses(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor
):
    """Reset object poses with random selection. Inactive objects moved far away to prevent tracking errors."""
    
    # Initialize active_objects tensor if it doesn't exist
    if not hasattr(env, 'active_objects'):
        env.active_objects = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Randomly select active object for each environment being reset
    env.active_objects[env_ids] = torch.randint(0, len(OBJECT_NAMES), (len(env_ids),), device=env.device)
    
    # Reset poses for all objects
    for obj_idx, obj_name in enumerate(OBJECT_NAMES):
        # Sample random pose for active objects
        object_pose = sample_root_state_uniform(
            env=env, env_ids=env_ids, 
            pose_range={'x': (-0.2, 0.2), 'y': (-0.2, 0.2), 'yaw': (0, 2 * math.pi)},
            asset_cfg=SceneEntityCfg(obj_name)
        )[0]
        
        # Determine which environments have this object active
        is_active = (env.active_objects[env_ids] == obj_idx)
        active_env_ids = env_ids[is_active]
        inactive_env_ids = env_ids[~is_active]
        
        # For active environments: set normal pose at table
        if len(active_env_ids) > 0:
            env.scene[obj_name].write_root_pose_to_sim(object_pose[is_active], env_ids=active_env_ids)
        
        # For inactive environments: move VERY far away at z=-100 so they CANNOT be tracked
        # This ensures observation functions will never accidentally get data from wrong objects
        if len(inactive_env_ids) > 0:
            # Create pose very far underground with zero velocity
            inactive_pose = object_pose[~is_active].clone()
            inactive_pose[:, 2] = -100.0  # z = -100 (extremely far, impossible to track)
            inactive_pose[:, 3:7] = 0.0   # zero linear velocity
            inactive_pose[:, 7:10] = 0.0  # zero angular velocity
            env.scene[obj_name].write_root_pose_to_sim(inactive_pose, env_ids=inactive_env_ids)


def clip_object_ranges(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor
):
    """Clip object ranges only for active objects."""
    # Handle None case
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    # Initialize active_objects tensor if it doesn't exist yet
    if not hasattr(env, 'active_objects'):
        env.active_objects = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Clip ranges for each object type
    for obj_idx, obj_name in enumerate(OBJECT_NAMES):
        # Find which environments have this object active
        is_active = (env.active_objects[env_ids] == obj_idx)
        active_env_ids = env_ids[is_active]
        
        if len(active_env_ids) > 0:
            clip_object_xy_range(
                env=env, env_ids=active_env_ids,
                x_range=(0.05, 0.55), y_range=(-0.3, 0.3), z_thresh=0.05,
                asset_cfg=SceneEntityCfg(obj_name)
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
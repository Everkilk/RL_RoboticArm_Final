# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP helper functions for Franka-Shadow robot manipulation tasks."""

import torch
import math
from typing import Tuple

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
    
##
# Helper Functions
##


def euler_to_quaternion(
    yaw: float, pitch: float, roll: float
) -> Tuple[float, float, float, float]:
    """
    Converts Euler angles to a quaternion.

    Args:
        yaw: Yaw angle in radians.
        pitch: Pitch angle in radians.
        roll: Roll angle in radians.

    Returns:
        A tuple representing the quaternion (w, x, y, z).
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (w, x, y, z)


def get_joint_pos_rel(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def get_joint_vel_rel(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]


def get_last_action(
    env: ManagerBasedRLEnv, 
    action_name: str | None = None
) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions


##
# Observation Helper Functions
##

def get_object_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get object pose (position + orientation) in robot root frame.
    
    Returns:
        Tensor of shape (num_envs, 6): [pos_x, pos_y, pos_z, euler_x, euler_y, euler_z]
    """
    pos, euler = get_object_pose_in_robot_root_frame(env)
    return torch.cat([pos, euler], dim=-1)


def get_hand_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get hand/palm pose (position + orientation) in robot root frame.
    
    Returns:
        Tensor of shape (num_envs, 6): [pos_x, pos_y, pos_z, euler_x, euler_y, euler_z]
    """
    # Get palm frame from frame transformer
    hand_frame = env.scene.sensors["base_hand_frame"]
    hand_pos = hand_frame.data.target_pos_w[:, 0, :]  # First target is palm
    hand_quat = hand_frame.data.target_quat_w[:, 0, :]  # First target is palm
    
    # Get robot root frame (source frame of base_hand_frame)
    robot_pos = hand_frame.data.source_pos_w
    robot_quat = hand_frame.data.source_quat_w
    
    # Transform hand pose to robot root frame
    hand_pos_rel = math_utils.subtract_frame_transforms(
        robot_pos, robot_quat, hand_pos, hand_quat
    )[0]
    
    # Convert quaternion to euler angles (returns tuple of 3 tensors)
    hand_euler = torch.stack(math_utils.euler_xyz_from_quat(hand_quat), dim=-1)
    
    # Concatenate position and orientation
    return torch.cat([hand_pos_rel, hand_euler], dim=-1)


def get_fingertip_poses(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get all 5 fingertip poses (position + orientation) in robot root frame.
    
    Returns:
        Tensor of shape (num_envs, 30): 5 fingertips * 6 values (pos + euler)
    """
    # Get fingertip frames from frame transformer
    fingertip_frame = env.scene.sensors["target_finger_frames"]
    fingertip_pos = fingertip_frame.data.target_pos_w  # (num_envs, 5, 3)
    fingertip_quat = fingertip_frame.data.target_quat_w  # (num_envs, 5, 4)
    
    # Get robot root frame (source frame)
    robot_pos = fingertip_frame.data.source_pos_w  # (num_envs, 3)
    robot_quat = fingertip_frame.data.source_quat_w  # (num_envs, 4)
    
    # Transform all fingertip poses to robot root frame
    num_envs = fingertip_pos.shape[0]
    fingertip_poses = []
    
    for i in range(5):  # 5 fingertips
        # Get position relative to robot root
        pos_rel = math_utils.subtract_frame_transforms(
            robot_pos, robot_quat, fingertip_pos[:, i, :], fingertip_quat[:, i, :]
        )[0]
        
        # Convert quaternion to euler angles (returns tuple of 3 tensors)
        euler = torch.stack(math_utils.euler_xyz_from_quat(fingertip_quat[:, i, :]), dim=-1)
        
        # Concatenate position and orientation
        fingertip_poses.append(torch.cat([pos_rel, euler], dim=-1))
    
    # Stack all fingertips
    return torch.cat(fingertip_poses, dim=-1)  # (num_envs, 30)


##
# Core MDP Functions
##

def get_relative_object_position(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | None=None,
    object_cfg: SceneEntityCfg=SceneEntityCfg('cube')
):
    """
    The relative position from the environment origin to the object
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]


def substract_transform_in_robot_root_frame(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | None=None,
    t: torch.Tensor | None=None, quat: torch.Tensor | None=None,
    robot_cfg: SceneEntityCfg=SceneEntityCfg('robot'),
):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene[robot_cfg.name]
    return math_utils.subtract_frame_transforms(
        t01=robot.data.root_pos_w[env_ids], q01=robot.data.root_quat_w[env_ids], t02=t, q02=quat
    )


def get_object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | None=None,
    robot_cfg: SceneEntityCfg=SceneEntityCfg('robot'),
    object_cfg: SceneEntityCfg=SceneEntityCfg('cube')
):
    """
    The position and orientation (Euler angles) of the object in the robot's root frame.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_b, object_quat_b = math_utils.subtract_frame_transforms(
        t01=robot.data.root_pos_w[env_ids], q01=robot.data.root_quat_w[env_ids],
        t02=object.data.root_pos_w[env_ids], q02=object.data.root_quat_w[env_ids]
    )
    object_euler_b = torch.stack(math_utils.euler_xyz_from_quat(object_quat_b), dim=1)
    return object_pos_b, object_euler_b  # (n, 3), (n, 3)


def get_object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | None=None,
    robot_cfg: SceneEntityCfg=SceneEntityCfg('robot'),
    object_cfg: SceneEntityCfg=SceneEntityCfg('cube')
):
    """
    The position of the object in the robot's root frame.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_b, _ = math_utils.subtract_frame_transforms(
        t01=robot.data.root_pos_w[env_ids], q01=robot.data.root_quat_w[env_ids], t02=object.data.root_pos_w[env_ids]
    )
    return object_pos_b


def get_object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | None=None,
    robot_cfg: SceneEntityCfg=SceneEntityCfg('robot'),
    object_cfg: SceneEntityCfg=SceneEntityCfg('cube')
):
    """
    The orientation (Euler angles) of the object in the robot's root frame.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    _, object_quat_b = math_utils.subtract_frame_transforms(
        t01=robot.data.root_pos_w[env_ids], q01=robot.data.root_quat_w[env_ids], q02=object.data.root_quat_w[env_ids]
    )
    object_euler_b = torch.stack(math_utils.euler_xyz_from_quat(object_quat_b), dim=1)
    return object_euler_b  # (n, 3)


def get_end_effector_positions_in_robot_root_frame(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | None=None,
    ee_frame_cfg=SceneEntityCfg("ee_frame")
):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    return env.scene[ee_frame_cfg.name].data.target_pos_source[env_ids]  # (n, b, 3)


def get_end_effector_orientations_in_robot_root_frame(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | None=None,
    ee_frame_cfg=SceneEntityCfg("ee_frame")
):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    quats = env.scene[ee_frame_cfg.name].data.target_quat_source[env_ids]  # (n, b, 4)
    return torch.stack(math_utils.euler_xyz_from_quat(quats.view(-1, 4)), dim=1).view(len(env_ids), -1, 3)


def check_invalid_object_range(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | None,
    x_range: tuple, y_range: tuple, z_thresh: float,
    asset_cfg: SceneEntityCfg=SceneEntityCfg("cube")
):
    obj_rel_pos_w = get_relative_object_position(env, env_ids, asset_cfg)
    return (
        ((obj_rel_pos_w[:, 0] < x_range[0] - 1e-4) |
         (obj_rel_pos_w[:, 0] > x_range[1] + 1e-4) |
         (obj_rel_pos_w[:, 1] < y_range[0] - 1e-4) |
         (obj_rel_pos_w[:, 1] > y_range[1] + 1e-4)) &
        (obj_rel_pos_w[:, 2] < z_thresh)
    )


def check_collisions(
    env: ManagerBasedRLEnv, 
    threshold: float=1.0,
    env_ids: torch.Tensor | None=None,
    contact_sensor_cfg: SceneEntityCfg=SceneEntityCfg('contact_sensor')
):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    return (env.scene[contact_sensor_cfg.name]._data.net_forces_w.norm(dim=-1) > threshold).any(dim=1, keepdim=True)


def check_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length


def clip_object_xy_range(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor | None,
    x_range: tuple, y_range: tuple, z_thresh: float,
    asset_cfg: SceneEntityCfg=SceneEntityCfg("object")
):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # get object position
    obj_pos_w = asset.data.root_pos_w[env_ids]
    # get environment origin
    origin = env.scene.env_origins[env_ids]

    # check if object is in the valid range
    env_ids = torch.where(
        ((obj_pos_w[:, 0] < origin[:, 0] + x_range[0] - 1e-4) |
         (obj_pos_w[:, 0] > origin[:, 0] + x_range[1] + 1e-4) |
         (obj_pos_w[:, 1] < origin[:, 1] + y_range[0] - 1e-4) |
         (obj_pos_w[:, 1] > origin[:, 1] + y_range[1] + 1e-4)) &
        (obj_pos_w[:, 2] < origin[:, 2] + z_thresh)
    )[0]

    if len(env_ids):
        # get object states from environments contained objects be not in the valid range.
        obj_pos_w = obj_pos_w[env_ids]
        origin = origin[env_ids]
        obj_quat_w = asset.data.root_quat_w[env_ids]

        # take object to the valid position
        obj_pos_w[:, 0] = torch.clamp(
            input=obj_pos_w[:, 0],
            min=origin[:, 0] + x_range[0],
            max=origin[:, 0] + x_range[1]
        )
        obj_pos_w[:, 1] = torch.clamp(
            input=obj_pos_w[:, 1],
            min=origin[:, 1] + y_range[0],
            max=origin[:, 1] + y_range[1]
        )
        # write new position to simulation
        asset.write_root_pose_to_sim(torch.cat([obj_pos_w, obj_quat_w], dim=-1), env_ids=env_ids)


def sample_root_state_uniform(    
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]]={},
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()
    
    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples
    return torch.cat([positions, orientations], dim=-1), velocities


def reset_scene_to_default(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Reset the scene to the default state specified in the scene configuration."""
    # rigid bodies
    for rigid_object in env.scene.rigid_objects.values():
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        rigid_object.write_root_state_to_sim(default_root_state, env_ids=env_ids)
    # articulations
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_state_to_sim(default_root_state, env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

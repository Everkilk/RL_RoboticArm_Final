import torch
from .common import check_time_out, ManagerBasedRLEnv
from .observation import get_command, get_object_position, get_object_orientation

# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.managers import TerminationTermCfg as DoneTerm

###
##### TERMINAL PART
###

def check_goal_achieved(env: ManagerBasedRLEnv, dis_thresh: float, angle_thresh: float):
    """Check if goal is achieved based on both position and orientation.
    
    Args:
        env: The environment
        dis_thresh: Position distance threshold in meters (e.g., 0.03)
        angle_thresh: Orientation angle threshold in degrees (e.g., 15.0)
    """
    target_pose = get_command(env)  # (n, 7): pos(3) + quat(4)
    target_pos = target_pose[:, :3]
    target_quat = target_pose[:, 3:]
    
    object_pos = get_object_position(env)  # (n, 3)
    object_orient = get_object_orientation(env)  # (n, 3) euler angles
    
    # Check position
    position_achieved = (target_pos - object_pos).norm(p=2, dim=-1) <= dis_thresh
    
    # Check orientation - convert object euler to quaternion and compare
    object_quat = math_utils.quat_from_euler_xyz(
        object_orient[:, 0], object_orient[:, 1], object_orient[:, 2]
    )
    
    # Calculate quaternion difference and extract angle
    quat_diff = math_utils.quat_mul(object_quat, math_utils.quat_conjugate(target_quat))
    angle_diff_rad = 2 * torch.acos(quat_diff[:, 0].clamp(-1.0, 1.0))
    angle_diff_deg = torch.rad2deg(angle_diff_rad)
    orientation_achieved = angle_diff_deg <= angle_thresh
    
    # Both conditions must be satisfied
    return position_achieved & orientation_achieved


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=check_time_out, time_out=True)

    goal_achieved = DoneTerm(func=check_goal_achieved, params={'dis_thresh': 0.03, 'angle_thresh': 15.0})
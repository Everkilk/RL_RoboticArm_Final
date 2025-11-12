from .common import *

# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

###
##### OBSERVATION FUNCTIONS
###

def get_object_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get object position in robot root frame."""
    return get_object_position_in_robot_root_frame(
        env=env, 
        robot_cfg=SceneEntityCfg('robot'),
        object_cfg=SceneEntityCfg('cube'),
    )  # (n, 3)

def get_object_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get object orientation in robot root frame."""
    return get_object_orientation_in_robot_root_frame(
        env=env, 
        robot_cfg=SceneEntityCfg('robot'),
        object_cfg=SceneEntityCfg('cube')
    )  # (n, 3) 

def get_object_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get object position and orientation in robot root frame."""
    return torch.cat([get_object_position(env), get_object_orientation(env)], dim=-1) # (n, 6)


def get_hand_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get hand position in robot root frame."""
    return get_end_effector_positions_in_robot_root_frame(
        env=env, ee_frame_cfg=SceneEntityCfg('target_hand_frame')
    ).view(-1, 3)  # (n, 3)

def get_hand_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get hand orientation in robot root frame."""
    return get_end_effector_orientations_in_robot_root_frame(
        env=env, ee_frame_cfg=SceneEntityCfg('target_hand_frame')
    ).view(-1, 3)  # (n, 3)

def get_hand_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get hand position and orientation in robot root frame."""
    return torch.cat([get_hand_position(env), get_hand_orientation(env)], dim=-1)  # (n, 6)


def get_fingertip_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get all fingertip positions."""
    return get_end_effector_positions_in_robot_root_frame(
        env=env, ee_frame_cfg=SceneEntityCfg('target_finger_frames')
    ).view(-1, 15)  # (n, 15)

def get_fingertip_orientations(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get all fingertip orientations."""
    return get_end_effector_orientations_in_robot_root_frame(
        env=env, ee_frame_cfg=SceneEntityCfg('target_finger_frames')
    ).view(-1, 15)  # (n, 15)

def get_fingertip_poses(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get all fingertip positions and orientations."""
    return torch.cat([
        get_fingertip_positions(env), 
        get_fingertip_orientations(env)
    ], dim=-1) # (n, 30)


def get_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The generated command from command term in the command manager."""
    return env.command_manager.get_command('object_pos')


def format_target_goals(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Format target goals for HER-style training (3 stages, 10 dims per stage).

    Goal layout per stage (10D): [hand_target_pos(3), object_target_pos(3), target_quat(4)]
      - Stage 0 (reach): hand_target_pos = object_pos, others zero/identity
      - Stage 1 (orientation): target_quat = identity (upright), others zero
      - Stage 2 (lift): object_target_pos = commanded target position, others zero/identity
    """
    device = env.device
    target_pos = get_command(env)  # (n, 3)
    object_pos = get_object_position(env)  # (n, 3)
    # Identity quaternion [1, 0, 0, 0] repeated
    id_quat = torch.zeros((len(object_pos), 4), device=object_pos.device, dtype=object_pos.dtype)
    id_quat[:, 0] = 1.0

    stage0 = torch.cat([object_pos, torch.zeros_like(target_pos), id_quat], dim=1)  # (n, 10)
    stage1 = torch.cat([torch.zeros_like(object_pos), torch.zeros_like(target_pos), id_quat], dim=1)  # (n, 10)
    stage2 = torch.cat([torch.zeros_like(object_pos), target_pos, id_quat], dim=1)  # (n, 10)

    return torch.stack([stage0, stage1, stage2], dim=1)  # (3, 10)


def format_achieved_goals(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Format achieved goals for HER-style training (3 stages, 10 dims per stage).

    Achieved layout per stage (10D): [ach_hand_pos(3), ach_object_pos(3), ach_object_quat(4)]
      - Stage 0 (reach): ach_hand_pos
      - Stage 1 (orientation): ach_object_quat
      - Stage 2 (lift): ach_object_pos
    """
    hand_pos = get_hand_position(env)  # (n, 3)
    object_pos = get_object_position(env)  # (n, 3)
    object_euler = get_object_orientation(env)  # (n, 3)
    object_quat = math_utils.quat_from_euler_xyz(object_euler[:, 0], object_euler[:, 1], object_euler[:, 2])  # (n,4)

    id_quat = torch.zeros((len(object_pos), 4), device=object_pos.device, dtype=object_pos.dtype)
    id_quat[:, 0] = 1.0

    stage0 = torch.cat([hand_pos, torch.zeros_like(object_pos), id_quat], dim=1)  # (n, 10)
    stage1 = torch.cat([torch.zeros_like(hand_pos), torch.zeros_like(object_pos), object_quat], dim=1)  # (n, 10)
    stage2 = torch.cat([torch.zeros_like(hand_pos), object_pos, id_quat], dim=1)  # (n, 10)

    return torch.stack([stage0, stage1, stage2], dim=1)  # (3, 10)


def get_invalid_hand(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check if hand is in invalid position (too low)."""
    hand_ee_frame = get_end_effector_positions_in_robot_root_frame(
        env=env, ee_frame_cfg=SceneEntityCfg('base_hand_frame')
    ).view(-1, 3)
    return (hand_ee_frame[:, 2] - 0.075 <= 0.0).view(-1, 1)

def get_invalid_object_range(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check if object is outside valid workspace range."""
    return check_invalid_object_range(
        env=env, env_ids=None,
        x_range=(0.1, 0.5), y_range=(-0.2, 0.2), z_thresh=0.05, 
        asset_cfg=SceneEntityCfg('cube')
    ).view(-1, 1)

def get_dangerous_robot_collisions(env: ManagerBasedRLEnv) -> torch.Tensor:
    return check_collisions(env, threshold=10.0, contact_sensor_cfg=SceneEntityCfg('contact_sensor'))


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ObservationCfg(ObsGroup):
        """Observations for policy group."""

        object_pose = ObsTerm(func=get_object_pose)  # (6,)
        hand_pose = ObsTerm(func=get_hand_pose)  # (6,)
        fingertip_poses = ObsTerm(func=get_fingertip_poses)  # (30,)
        joint_pos = ObsTerm(func=get_joint_pos_rel)  # (31,) - 7 arm + 24 hand
        joint_vel = ObsTerm(func=get_joint_vel_rel)  # (31,)
        last_action = ObsTerm(func=get_last_action)  # (25,) - 6 IK + 19 finger

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class DesiredGoalCfg(ObsGroup):
        """Desired goal for the policy."""
        goals = ObsTerm(func=format_target_goals)

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class AchievedGoalCfg(ObsGroup):
        """Achieved goal for the policy."""
        goals = ObsTerm(func=format_achieved_goals)

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class MetaCfg(ObsGroup):
        """Meta information for validity checking."""
        invalid_hand = ObsTerm(func=get_invalid_hand)  # (1,)
        invalid_object_range = ObsTerm(func=get_invalid_object_range)  # (1,)
        collisions = ObsTerm(func=get_dangerous_robot_collisions) # (1,)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups
    observation: ObservationCfg = ObservationCfg()
    desired_goal: DesiredGoalCfg = DesiredGoalCfg()
    achieved_goal: AchievedGoalCfg = AchievedGoalCfg()
    meta: MetaCfg = MetaCfg()
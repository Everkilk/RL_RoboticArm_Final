from .common import *

# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

###
##### OBSERVATION FUNCTIONS
###

# Object names matching events.py
OBJECT_NAMES = ['object_cube', 'object_mustard']  # 'object_drill' commented out in env_cfg

def _get_active_object_data(env: ManagerBasedRLEnv, data_type: str) -> torch.Tensor:
    """
    Helper function to get data from the currently active object for each environment.
    
    Args:
        env: The environment instance
        data_type: Type of data to retrieve ('position', 'orientation')
    
    Returns:
        Tensor of shape (num_envs, 3) containing the requested data
    """
    num_envs = env.num_envs
    device = env.device
    
    # Initialize active_objects tensor if it doesn't exist yet
    if not hasattr(env, 'active_objects'):
        env.active_objects = torch.zeros(num_envs, dtype=torch.long, device=device)
    
    # Initialize output tensor
    if data_type == 'position':
        output = torch.zeros((num_envs, 3), device=device)
        get_func = get_object_position_in_robot_root_frame
    elif data_type == 'orientation':
        output = torch.zeros((num_envs, 3), device=device)
        get_func = get_object_orientation_in_robot_root_frame
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    # For each object type, get data for environments where it's active
    for obj_idx, obj_name in enumerate(OBJECT_NAMES):
        # Find which environments have this object active
        is_active = (env.active_objects == obj_idx)
        
        if is_active.any():
            # Get data for this object
            obj_data = get_func(
                env=env,
                robot_cfg=SceneEntityCfg('robot'),
                object_cfg=SceneEntityCfg(obj_name)
            )  # (num_envs, 3)
            
            # Copy only for active environments
            output[is_active] = obj_data[is_active]
    
    return output


def get_object_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get active object position in robot root frame."""
    return _get_active_object_data(env, 'position')  # (n, 3)


def get_object_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get active object orientation in robot root frame."""
    return _get_active_object_data(env, 'orientation')  # (n, 3) 

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
    """Format target goals for HER-style training."""
    target_pos = get_command(env) # (n, 3)
    object_pos = get_object_position(env) # (n, 3)
    return torch.stack([
        torch.cat([object_pos, torch.zeros_like(target_pos)], dim=1),
        torch.cat([torch.zeros_like(object_pos), target_pos], dim=1)
    ], dim=1)  # (2, 6)


def format_achieved_goals(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Format achieved goals for HER-style training."""
    hand_pos = get_hand_position(env)  # (n, 3)
    object_pos = get_object_position(env) # (n, 3)
    return torch.stack([
        torch.cat([hand_pos, torch.zeros_like(object_pos)], dim=1),
        torch.cat([torch.zeros_like(hand_pos), object_pos], dim=1)
    ], dim=1)  # (2, 6)


def get_invalid_hand(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check if hand is in invalid position (too low)."""
    hand_ee_frame = get_end_effector_positions_in_robot_root_frame(
        env=env, ee_frame_cfg=SceneEntityCfg('base_hand_frame')
    ).view(-1, 3)
    return (hand_ee_frame[:, 2] - 0.075 <= 0.0).view(-1, 1)

def get_invalid_object_range(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check if active object is outside valid workspace range."""
    num_envs = env.num_envs
    device = env.device
    
    # Initialize active_objects tensor if it doesn't exist yet
    if not hasattr(env, 'active_objects'):
        env.active_objects = torch.zeros(num_envs, dtype=torch.long, device=device)
    
    output = torch.zeros((num_envs, 1), device=device, dtype=torch.bool)
    
    # Check each object type for environments where it's active
    for obj_idx, obj_name in enumerate(OBJECT_NAMES):
        is_active = (env.active_objects == obj_idx)
        
        if is_active.any():
            obj_invalid = check_invalid_object_range(
                env=env, env_ids=None,
                x_range=(0.1, 0.5), y_range=(-0.2, 0.2), z_thresh=0.05, 
                asset_cfg=SceneEntityCfg(obj_name)
            ).view(-1, 1)
            
            output[is_active] = obj_invalid[is_active]
    
    return output

def get_dangerous_robot_collisions(env: ManagerBasedRLEnv) -> torch.Tensor:
    return check_collisions(env, threshold=10.0, contact_sensor_cfg=SceneEntityCfg('contact_sensor'))

def get_object_class_one_hot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get one-hot encoding of the active object class."""
    num_envs = env.num_envs
    device = env.device
    num_classes = len(OBJECT_NAMES)
    
    # Initialize active_objects tensor if it doesn't exist yet
    if not hasattr(env, 'active_objects'):
        env.active_objects = torch.zeros(num_envs, dtype=torch.long, device=device)
    
    one_hot = torch.zeros((num_envs, num_classes), device=device)
    one_hot.scatter_(1, env.active_objects.view(-1, 1), 1.0)
    
    return one_hot  # (n, num_classes)


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
        object_class = ObsTerm(func=get_object_class_one_hot)  # (2,) one-hot for 2 objects

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
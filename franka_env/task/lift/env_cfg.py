# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based environment configuration for Franka-Shadow robot lift task."""

import sys
import math
from pathlib import Path
from omegaconf import MISSING
import gymnasium as gym

# Add the source directory to Python path
source_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))

from assets.Robots.franka_hand import FRANKA_SHADOW_CFG
print(f"[DEBUG] Loading Franka-Shadow robot from: {Path(__file__).parent.parent.parent}")
print(f"[DEBUG] Robot USD path: {FRANKA_SHADOW_CFG.spawn.usd_path}")

# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg, ContactSensorCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sim.spawners.shapes import SphereCfg, CuboidCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.lights import DomeLightCfg 
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Import MDP Configurations
from .mdp.common import *
from .mdp.observation import ObservationsCfg
from .mdp.action import ActionsCfg
from .mdp.reward import FrankaCudeLiftReward
from .mdp.command import CommandsCfg
from .mdp.termination import TerminationsCfg
from .mdp.events import EventsCfg

######################################################################################################
######################################### SCENE DEFINITION ###########################################
######################################################################################################

@configclass
class FrankaShadowLiftSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with Franka-Shadow robot and object."""

    # Robot configuration
    robot: ArticulationCfg = FRANKA_SHADOW_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    # contact and end-effector sensors: will be populated by agent env cfg
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/robot/.*", 
        update_period=0.0, 
        history_length=6, 
        debug_vis=False,
        force_threshold=10.0
    )
    
    # Base hand frame - tracks the robot base for reference
    base_hand_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/fr3_link0",
        debug_vis=False,  # Disabled to avoid errors during initialization
        visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/World/visuals/base_hand_marker",
            markers={
                "frame": SphereCfg(
                    radius=0.075,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                )
            }
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/robot0_palm",
                name="end_effector_hand_target",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.12],
                    rot=euler_to_quaternion(0.0, 0.0, 0.0)
                )
            )
        ]
    )
    
    # Target hand frame - tracks the palm position with offset for grasping
    target_hand_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/fr3_link0",
        debug_vis=False,  # Disabled to avoid errors during initialization
        visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/World/visuals/target_hand_marker",
            markers={
                "frame": CuboidCfg(
                    size=[0.12, 0.12, 0.09],
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                )
            }
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/robot0_palm",
                name="end_effector_hand_target",
                offset=OffsetCfg(
                    pos=[0.0, -0.07, 0.09],
                    rot=euler_to_quaternion(0.0, 0.0, (75 / 180) * math.pi)
                )
            ),
        ]
    )

    # Fingertip frames - tracks all 5 fingertips for contact detection
    target_finger_frames: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/fr3_link0",  # Use a rigid body link as source
        debug_vis=False,  # Disabled to avoid errors during initialization
        visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/World/visuals/finger_marker",
            markers={
                "frame": SphereCfg(
                    radius=0.01,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                )
            }
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/robot0_thdistal",
                name="end_effector_thumb",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.025]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/robot0_ffdistal",
                name="end_effector_index_finger",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.025]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/robot0_mfdistal",
                name="end_effector_middle_finger",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.025]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/robot0_rfdistal",
                name="end_effector_ring_finger",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.025]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/robot0_lfdistal",
                name="end_effector_pinky",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.025]),
            ),
        ],  
    )
    
    # Object 1: DexCube 
    object_cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object_cube",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0.0, 0.08), rot=(1.0, 0.0, 0.0, 0.0)  # Higher z to prevent falling through table
        ),
    )
    
    # Object 2: YCB Mustard Bottle
    object_mustard: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object_mustard",
        spawn=UsdFileCfg(
            usd_path=str(Path(__file__).parent.parent.parent / "assets" / "Object" / "mustard_bottle.usd"),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=MassPropertiesCfg(mass=0.44),  # 440 grams
            scale=(0.6, 0.6, 0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0.0, 0.05), rot=(0.7071, 0.7071, 0.0, 0.0)  # 90° rotation around X-axis to stand up
        ),
    )
    
    # # Object 3: YCB Drill
    # object_drill: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/object_drill",
    #     spawn=UsdFileCfg(
    #         usd_path=str(Path(__file__).parent.parent.parent / "assets" / "Object" / "drill.usd"),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=8,
    #             solver_velocity_iteration_count=0,
    #             sleep_threshold=0.005,
    #             stabilization_threshold=0.0025,
    #             max_depenetration_velocity=1000.0,
    #         ),
    #         mass_props=MassPropertiesCfg(mass=1.2),  # 1200 grams
    #         scale=(1.0, 1.0, 1.0),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.3, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0)
    #     ),
    # )

    # Environment objects
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/ground_plane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


######################################################################################################
######################################### ENVIRONMENT CONFIG #########################################
######################################################################################################

@configclass
class FrankaShadowLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Franka-Shadow lift task using manager-based approach."""
    
    # Scene settings
    scene: FrankaShadowLiftSceneCfg = FrankaShadowLiftSceneCfg(num_envs=16, env_spacing=2.5)
    
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # Note: Rewards not used in goal-conditioned RL (using custom reward_func instead)
    # But required by ManagerBasedRLEnvCfg for validation
    rewards: dict = {}  # Empty dict as placeholder
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    
    # Environment settings
    decimation = 2
    episode_length_s = 10.0
    seed = None  # Will be set from command line args
    
    # Gym space settings (required for Isaac Lab 2.2+)
    # Isaac Lab 2.2.1 requires explicit space definitions (None not allowed)
    # Observation space structure (matching ObservationsCfg):
    # - observation: Box(134,) - concatenated obs (6+6+30+31+31+25=129) + padding = 134
    # - desired_goal: Box(6,) - target object pose
    # - achieved_goal: Box(6,) - current object pose  
    # - meta: Box(3,) - validity flags (invalid_hand, invalid_range, collisions)
    observation_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(134,)),
        "desired_goal": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(6,)),
        "achieved_goal": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(6,)),
        "meta": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,))
    })
    # Action space: Box(30,) - 6 IK commands (DifferentialIK) + 24 finger joints (Shadow Hand)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(30,))
    state_space = None  # For asymmetric actor-critic (not used here)
    num_states = None  # Optional: for asymmetric actor-critic
    # Deprecated attributes (kept for backward compatibility with Isaac Lab API check)
    num_actions = 30  # 6 IK commands + 24 finger joints
    num_observations = 136  # Robot state observations (updated for multi-object: object_pose=6, hand_pose=6, fingertip_poses=30, joint_pos=31, joint_vel=31, last_action=25, total=129+6+1=136)
    
    # Noise models (Isaac Lab 2.2+ requirement)
    action_noise_model = None  # No noise model by default
    observation_noise_model = None  # No observation noise by default
    
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 3.2  
        # simulation settings
        self.sim.dt = 0.025
        self.sim.render_interval = self.decimation
        # max_episode_length = episode_length_s / (dt * decimation) (steps)
        # select_action_frequency (from rl agent)= 1 / (dt * decimation) (Hz)
        # frame_per_second (for simulation) = 1 / dt (Hz)

        # physics settings
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**21 # Reduced by factor 2**(-4)
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**17
        self.sim.physx.gpu_max_rigid_contact_count = 2**19
        self.sim.physx.gpu_max_rigid_patch_count = 2**13
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**17
        self.sim.physx.gpu_collision_stack_size = 2**22
        self.sim.physx.gpu_heap_capacity = 2**22
        self.sim.physx.gpu_temp_buffer_capacity = 2**20
        self.sim.physx.gpu_max_soft_body_contacts = 2**16
        self.sim.physx.gpu_max_particle_contacts = 2**16
        
        # rl settings
        self.num_frames = 3
        # Note: num_actions and num_observations removed (deprecated in Isaac Lab 2.2+)
        # Spaces are now auto-configured from observations/actions managers
        self.num_goals = 6
        self.num_stages = 2
        self.reward_func = FrankaCudeLiftReward(scale_factor=1.0)

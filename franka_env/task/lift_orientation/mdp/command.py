from typing import Sequence
import torch
from .common import *

# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

###
##### COMMAND PART
###

class PositionCommand(CommandTerm):
    """Configuration for the command generator."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.target_ee = env.scene[cfg.target_ee_name]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        self.task_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # -- metrics
        self.pre_pos_object_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.pre_orient_object_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics['distance'] = torch.zeros(self.num_envs, device=self.device)
        self.metrics['moved_dist'] = torch.zeros(self.num_envs, device=self.device)
        self.metrics['moved_z'] = torch.zeros(self.num_envs, device=self.device)
        self.metrics['hand2obj_dist'] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> Sequence[torch.Tensor]:
        """The desired pose command. Shape is (num_envs, 6).

        The first three elements correspond to the position, followed by the euler angle orientation in (x, y, z).
        """
        return self.pose_command_b[:, :3]

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = math_utils.combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        
        # get current command object pose --------
        pos_command_b = self.command
        pos_object_b = get_object_position_in_robot_root_frame(
            env=self._env, robot_cfg=SceneEntityCfg(self.cfg.asset_name), object_cfg=SceneEntityCfg(self.cfg.object_name)
        )
        orient_object_b = get_object_orientation_in_robot_root_frame(
            env=self._env, robot_cfg=SceneEntityCfg(self.cfg.asset_name), object_cfg=SceneEntityCfg(self.cfg.object_name)
        )
        hand_ee_pos_b = get_hand_pose(env=self._env)[:, :3]
        # update metrics --------------------------
        self.metrics['distance'] = (pos_object_b - pos_command_b).norm(dim=1)
        self.metrics['moved_dist'] += (pos_object_b - self.pre_pos_object_b).norm(dim=1)
        self.metrics['moved_z'] += (pos_object_b[:, 2] - self.pre_pos_object_b[:, 2]).abs()
        self.metrics['hand2obj_dist'] = (hand_ee_pos_b - pos_object_b).norm(dim=-1)
        self.pre_pos_object_b = pos_object_b
        self.pre_orient_object_b = orient_object_b

    def _resample_command(self, env_ids: Sequence[int]):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.LongTensor(env_ids).to(self.device)
        
        # setup new pose targets -----------------
        self.pose_command_b[env_ids, 0] = self.pose_command_b[env_ids, 0].uniform_(0.2, 0.4)
        self.pose_command_b[env_ids, 1] = self.pose_command_b[env_ids, 1].uniform_(-0.15, 0.15)
        self.pose_command_b[env_ids, 2] = self.pose_command_b[env_ids, 1].uniform_(0.2, 0.35)

        # get object positions and orientation
        self.pre_pos_object_b[env_ids] = get_object_position_in_robot_root_frame(
            env=self._env, env_ids=env_ids,
            robot_cfg=SceneEntityCfg('robot'), 
            object_cfg=SceneEntityCfg('cube')
        )
        self.pre_orient_object_b[env_ids] = get_object_orientation_in_robot_root_frame(
            env=self._env, env_ids=env_ids, 
            robot_cfg=SceneEntityCfg('robot'), 
            object_cfg=SceneEntityCfg('cube')
        )
        
        for metric_name in self.metrics:
            self.metrics[metric_name][env_ids] = 0.0

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_pose"
                self.goal_pose_visualizer = VisualizationMarkers(marker_cfg)
                # -- current body pose
                marker_cfg.prim_path = "/Visuals/Command/body_pose"
                self.body_pose_visualizer = VisualizationMarkers(marker_cfg)
                # --- object pose
                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/object_pose"
                self.object_pose_visualizer = VisualizationMarkers(marker_cfg)
                # --- end effector pose
                marker_cfg.prim_path = "/Visuals/Command/ee_pose"
                self.ee_pose_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.body_pose_visualizer.set_visibility(True)
            self.object_pose_visualizer.set_visibility(True)
            self.ee_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.body_pose_visualizer.set_visibility(False)
                self.object_pose_visualizer.set_visibility(False)
                self.ee_pose_visualizer.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.body_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])
        # --- current object pose
        self.object_pose_visualizer.visualize(self.object.data.root_pos_w, self.object.data.root_quat_w)
        # --- current end effector pose
        self.ee_pose_visualizer.visualize(self.target_ee.data.target_pos_w.view(-1, 3), self.target_ee.data.target_quat_w.view(-1, 4))


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    
    @configclass
    class PositionCommandCfg(CommandTermCfg):
        """Configuration for uniform position command generator."""

        class_type: type = PositionCommand

        asset_name: str = 'robot'
        """Name of the asset in the environment for which the commands are generated."""
        
        body_name: str = 'fr3_link0'
        """Name of the body in the asset for which the commands are generated."""

        object_name: str = 'cube'
        """Name of the object in the environment for which the object are interacted."""
        
        target_ee_name: str = 'target_hand_frame'
        """Name of the target end-effector frame of the asset"""
        
        make_quat_unique: bool = False
        """Whether to make the quaternion unique or not. Defaults to False.

        If True, the quaternion is made unique by ensuring the real part is positive.
        """
    
    object_pos = PositionCommandCfg(
        resampling_time_range=(3.2, 3.2),
        debug_vis=True
    )
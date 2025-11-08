import math
import torch

# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg, 
    EMAJointPositionToLimitsActionCfg
)
from isaaclab.envs.mdp.actions import EMAJointPositionToLimitsAction

class FixedEMAJointPositionToLimitsAction(EMAJointPositionToLimitsAction):
    """Fixed version of EMAJointPositionToLimitsAction that handles shape correctly."""
    
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the internals with proper shape handling."""
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        
        # Get joint positions for the specified joints
        # Use proper tensor indexing: first get all joint data, then select joints
        if isinstance(env_ids, slice):
            joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        else:
            joint_pos = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
        
        # Handle different shapes that might come from Isaac Sim
        # joint_pos might be [num_envs, num_joints] or [num_envs, 1, num_joints]
        if joint_pos.ndim == 3:
            joint_pos = joint_pos.squeeze(1)
        
        # Set previous applied actions
        if isinstance(env_ids, slice):
            self._prev_applied_actions[:] = joint_pos
        else:
            self._prev_applied_actions[env_ids] = joint_pos

@configclass
class FixedEMAJointPositionToLimitsActionCfg(EMAJointPositionToLimitsActionCfg):
    """Configuration for fixed EMA joint position action."""
    
    class_type: type = FixedEMAJointPositionToLimitsAction
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Differential IK for Franka arm (6 DOF: 3 pos + 3 rot)
    arm_action: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["fr3_joint[1-7]"],  # Updated to match your USD file
        body_name="robot0_palm",
        controller=DifferentialIKControllerCfg(
            command_type="pose", 
            use_relative_mode=True, 
            ik_method="dls"
        ),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        scale=[0.1, 0.1, 0.1, math.pi / 6, math.pi / 6, math.pi / 6]
    )
    
    # Direct joint control for Shadow Hand (24 DOF)
    # Order matches robot joint order from debug output
    finger_action: FixedEMAJointPositionToLimitsActionCfg = FixedEMAJointPositionToLimitsActionCfg(
        asset_name="robot", 
        joint_names=[
            # Wrist (2 DOF)
            'robot0_WRJ1', 'robot0_WRJ0',
            # Fingers metacarpal joints (4 DOF)
            'robot0_FFJ3', 'robot0_MFJ3', 'robot0_RFJ3', 'robot0_LFJ4',
            # Thumb base (1 DOF)
            'robot0_THJ4',
            # Fingers proximal joints (4 DOF)
            'robot0_FFJ2', 'robot0_MFJ2', 'robot0_RFJ2', 'robot0_LFJ3',
            # Thumb proximal (1 DOF)
            'robot0_THJ3',
            # Fingers middle joints (4 DOF)
            'robot0_FFJ1', 'robot0_MFJ1', 'robot0_RFJ1', 'robot0_LFJ2',
            # Thumb middle (1 DOF)
            'robot0_THJ2',
            # Fingers distal joints (4 DOF)
            'robot0_FFJ0', 'robot0_MFJ0', 'robot0_RFJ0', 'robot0_LFJ1',
            # Thumb distal (2 DOF)
            'robot0_THJ1', 'robot0_LFJ0', 'robot0_THJ0',
        ],
        alpha=0.95
    )

import math

# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg, 
    EMAJointPositionToLimitsActionCfg
)
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
        scale=[0.05, 0.05, 0.05, math.pi / 12, math.pi / 12, math.pi / 12]
    )
    
    # Direct joint control for Shadow Hand fingers (19 DOF - excluding wrist)
    finger_action: EMAJointPositionToLimitsActionCfg = EMAJointPositionToLimitsActionCfg(
        asset_name="robot", 
        joint_names=[
            # Index finger (4 DOF)
            'robot0_FFJ3', 'robot0_FFJ2', 'robot0_FFJ1', 'robot0_FFJ0',
            # Middle finger (4 DOF)
            'robot0_MFJ3', 'robot0_MFJ2', 'robot0_MFJ1', 'robot0_MFJ0',
            # Ring finger (4 DOF)
            'robot0_RFJ3', 'robot0_RFJ2', 'robot0_RFJ1', 'robot0_RFJ0',
            # Little finger (5 DOF)
            'robot0_LFJ4', 'robot0_LFJ3', 'robot0_LFJ2', 'robot0_LFJ1', 'robot0_LFJ0',
            # Thumb (5 DOF - excluding wrist joints)
            'robot0_THJ4', 'robot0_THJ3', 'robot0_THJ2', 'robot0_THJ1', 'robot0_THJ0',
        ],
        alpha=0.95
    )

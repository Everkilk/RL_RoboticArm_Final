# Copyright (c) 2024-2025, Franka Shadow Hand Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka-Shadow robot (Franka Panda + Shadow Dexterous Hand).

This configuration combines:
- Franka Emika Panda: 7-DOF robotic arm
- Shadow Dexterous Hand: 24-DOF highly articulated hand

The robots are assembled with:
- Attachment point: panda_hand_joint (panda_link7) -> robot0:hand_mount
- Disabled joints: panda_finger_joint1, panda_finger_joint2

The following configurations are available:

* :obj:`FRANKA_SHADOW_CFG`: Franka-Shadow robot configuration with 
  7-DOF arm and 24-DOF dexterous hand (31 total DOF).
"""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Path to the USD file
ASSETS_DATA_DIR = Path(__file__).resolve().parents[2] / "assets"
# Print the USD path for debugging

usd_file_path = f"{ASSETS_DATA_DIR}/Robots/Franka_robot.usd"
print(f"[DEBUG] Loading Franka-Shadow assembled robot from: {usd_file_path}")
print(f"[DEBUG] File exists: {os.path.exists(usd_file_path)}")

###
###### ROBOT CONFIGURATION
###

"""
Configuration of Franka-Shadow robot.

Franka Panda 7-DOF arm + Shadow Hand 24-DOF (31 total actuated joints).
"""
FRANKA_SHADOW_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_file_path,
        activate_contact_sensors=True,  # Enable for hand grasping detection
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,  # Fix the base to prevent it from moving
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Robot base positioned for optimal reach
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        # Franka Panda arm joints - neutral ready pose
        joint_pos={
            # Franka Panda Arm (7-DOF)
            "fr3_joint1": 0.0,      # Base rotation
            "fr3_joint2": -0.569,   # Shoulder (about -32.6°)
            "fr3_joint3": 0.0,      # Upper arm rotation
            "fr3_joint4": -2.810,   # Elbow (about -161°)
            "fr3_joint5": 0.0,      # Forearm rotation
            "fr3_joint6": 3.037,    # Wrist 1 (about 174°)
            "fr3_joint7": 0.785,    # Wrist 2 (about 45°)

            # Shadow Hand - Wrist (2 DOF)
            "robot0_WRJ1": 0.0,   # Wrist radial/ulnar deviation
            "robot0_WRJ0": 0.0,   # Wrist flexion/extension
            
            # Shadow Hand - First Finger/Index (4 DOF)
            "robot0_FFJ3": 0.0,   # Index proximal
            "robot0_FFJ2": 0.0,   # Index middle
            "robot0_FFJ1": 0.0,   # Index distal
            "robot0_FFJ0": 0.0,   # Index tip
            
            # Shadow Hand - Middle Finger (4 DOF)
            "robot0_MFJ3": 0.0,   # Middle proximal
            "robot0_MFJ2": 0.0,   # Middle middle
            "robot0_MFJ1": 0.0,   # Middle distal
            "robot0_MFJ0": 0.0,   # Middle tip
            
            # Shadow Hand - Ring Finger (4 DOF)
            "robot0_RFJ3": 0.0,   # Ring proximal
            "robot0_RFJ2": 0.0,   # Ring middle
            "robot0_RFJ1": 0.0,   # Ring distal
            "robot0_RFJ0": 0.0,   # Ring tip
            
            # Shadow Hand - Little Finger (5 DOF)
            "robot0_LFJ4": 0.0,   # Little abduction
            "robot0_LFJ3": 0.0,   # Little proximal
            "robot0_LFJ2": 0.0,   # Little middle
            "robot0_LFJ1": 0.0,   # Little distal
            "robot0_LFJ0": 0.0,   # Little tip
            
            # Shadow Hand - Thumb (5 DOF)
            "robot0_THJ4": 0.0,   # Thumb abduction
            "robot0_THJ3": 0.0,   # Thumb proximal
            "robot0_THJ2": 0.0,   # Thumb middle
            "robot0_THJ1": 0.0,   # Thumb distal
            "robot0_THJ0": 0.0,   # Thumb tip
        },
        # Set initial joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Franka Panda Arm Actuators (7-DOF)
        # High-precision collaborative robot with excellent torque control
        "panda_arm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-7]"],
            effort_limit={
                "fr3_joint1": 87.0,   # Base - highest torque
                "fr3_joint2": 87.0,   # Shoulder
                "fr3_joint3": 87.0,   # Upper arm rotation
                "fr3_joint4": 87.0,   # Elbow
                "fr3_joint5": 12.0,   # Forearm rotation - lower torque
                "fr3_joint6": 12.0,   # Wrist 1
                "fr3_joint7": 12.0,   # Wrist 2
            },
            velocity_limit={
                "fr3_joint1": 2.175,  # ~124.5 deg/s
                "fr3_joint2": 2.175,
                "fr3_joint3": 2.175,
                "fr3_joint4": 2.175,
                "fr3_joint5": 2.610,  # ~149.5 deg/s
                "fr3_joint6": 2.610,
                "fr3_joint7": 2.610,
            },
            stiffness={
                "fr3_joint1": 400.0,
                "fr3_joint2": 400.0,
                "fr3_joint3": 400.0,
                "fr3_joint4": 400.0,
                "fr3_joint5": 400.0,
                "fr3_joint6": 400.0,
                "fr3_joint7": 400.0,
            },
            damping={
                "fr3_joint1": 80.0,
                "fr3_joint2": 80.0,
                "fr3_joint3": 80.0,
                "fr3_joint4": 80.0,
                "fr3_joint5": 80.0,
                "fr3_joint6": 80.0,
                "fr3_joint7": 80.0,
            },
        ),
        
        # Shadow Hand Actuators (24-DOF)
        # Highly dexterous hand with tendon-driven actuation
        "shadow_hand_wrist": ImplicitActuatorCfg(
            joint_names_expr=["robot0_WRJ.*"],
            effort_limit={
                "robot0_WRJ1": 10.0,  # WRJ1
                "robot0_WRJ0": 10.0,  # WRJ0 (if exists, was WRJ2 in docs)
            },
            velocity_limit=2.0,
            stiffness={
                "robot0_WRJ1": 1200.0,
                "robot0_WRJ0": 1200.0,
            },
            damping={
                "robot0_WRJ1": 40.0,
                "robot0_WRJ0": 40.0,
            },
        ),
        
        "shadow_hand_thumb": ImplicitActuatorCfg(
            joint_names_expr=["robot0_THJ.*"],
            effort_limit={
                "robot0_THJ4": 2.0,
                "robot0_THJ3": 1.0,
                "robot0_THJ2": 1.0,
                "robot0_THJ1": 1.0,
                "robot0_THJ0": 1.0,
            },
            velocity_limit=2.0,
            stiffness={
                "robot0_THJ4": 180.0,
                "robot0_THJ3": 220.0,
                "robot0_THJ2": 300.0,
                "robot0_THJ1": 350.0,
                "robot0_THJ0": 400.0,
            },
            damping={
                "robot0_THJ4": 6.0,
                "robot0_THJ3": 9.0,
                "robot0_THJ2": 12.0,
                "robot0_THJ1": 14.0,
                "robot0_THJ0": 16.0,
            },
        ),
        
        "shadow_hand_first_finger": ImplicitActuatorCfg(
            joint_names_expr=["robot0_FFJ.*"],
            effort_limit={
                "robot0_FFJ3": 1.0,
                "robot0_FFJ2": 1.0,
                "robot0_FFJ1": 0.5,
                "robot0_FFJ0": 0.5,
            },
            velocity_limit=2.0,
            stiffness={
                "robot0_FFJ3": 150.0,
                "robot0_FFJ2": 200.0,
                "robot0_FFJ1": 250.0,
                "robot0_FFJ0": 300.0,
            },
            damping={
                "robot0_FFJ3": 5.0,
                "robot0_FFJ2": 8.0,
                "robot0_FFJ1": 10.0,
                "robot0_FFJ0": 12.0,
            },
        ),
        
        "shadow_hand_middle_finger": ImplicitActuatorCfg(
            joint_names_expr=["robot0_MFJ.*"],
            effort_limit={
                "robot0_MFJ3": 1.0,
                "robot0_MFJ2": 1.0,
                "robot0_MFJ1": 0.5,
                "robot0_MFJ0": 0.5,
            },
            velocity_limit=2.0,
            stiffness={
                "robot0_MFJ3": 150.0,
                "robot0_MFJ2": 200.0,
                "robot0_MFJ1": 250.0,
                "robot0_MFJ0": 300.0,
            },
            damping={
                "robot0_MFJ3": 5.0,
                "robot0_MFJ2": 8.0,
                "robot0_MFJ1": 10.0,
                "robot0_MFJ0": 12.0,
            },
        ),
        
        "shadow_hand_ring_finger": ImplicitActuatorCfg(
            joint_names_expr=["robot0_RFJ.*"],
            effort_limit={
                "robot0_RFJ3": 1.0,
                "robot0_RFJ2": 1.0,
                "robot0_RFJ1": 0.5,
                "robot0_RFJ0": 0.5,
            },
            velocity_limit=2.0,
            stiffness={
                "robot0_RFJ3": 150.0,
                "robot0_RFJ2": 200.0,
                "robot0_RFJ1": 250.0,
                "robot0_RFJ0": 300.0,
            },
            damping={
                "robot0_RFJ3": 5.0,
                "robot0_RFJ2": 8.0,
                "robot0_RFJ1": 10.0,
                "robot0_RFJ0": 12.0,
            },
        ),
        
        "shadow_hand_little_finger": ImplicitActuatorCfg(
            joint_names_expr=["robot0_LFJ.*"],
            effort_limit={
                "robot0_LFJ4": 1.0,
                "robot0_LFJ3": 1.0,
                "robot0_LFJ2": 1.0,
                "robot0_LFJ1": 0.5,
                "robot0_LFJ0": 0.5,
            },
            velocity_limit=2.0,
            stiffness={
                "robot0_LFJ4": 200.0,
                "robot0_LFJ3": 200.0,
                "robot0_LFJ2": 250.0,
                "robot0_LFJ1": 300.0,
                "robot0_LFJ0": 350.0,
            },
            damping={
                "robot0_LFJ4": 8.0,
                "robot0_LFJ3": 8.0,
                "robot0_LFJ2": 10.0,
                "robot0_LFJ1": 12.0,
                "robot0_LFJ0": 14.0,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
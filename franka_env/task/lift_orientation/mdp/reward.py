from typing import Optional, Tuple
import torch

import isaaclab.utils.math as math_utils

###
##### REWARD FUNCTION
###

class FrankaCudeLiftReward:
    """Custom reward function for Franka-Shadow cube lifting + orientation task.

    Stages:
        0: Reach object (fingertips approach object volume)
        1: Align object orientation with goal orientation (dense)
        2: Lift object to goal position
    """

    def __init__(self, scale_factor: float = 1.0, orient_reward_scale: float = 1.0):
        self.scale_factor = scale_factor
        self.object_lengths = torch.tensor([0.06, 0.06, 0.06])
        self.grasp_range = torch.tensor([0.12, 0.12, 0.09])
        self.orient_reward_scale = orient_reward_scale
    
    def __call__(
        self, 
        next_observations: torch.Tensor, 
        encoded_goals: torch.Tensor, 
        stage_id: int, 
        metas: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rewards for the lift task.
        
        Args:
            next_observations: Current and next observations
            encoded_goals: Target encoded goals
            stage_id: Current stage (0=reach, 1=orient, 2=lift)
            metas: Meta information for penalties
        """
        # Pre-process
        if isinstance(next_observations, (list, tuple)):
            next_observations = next_observations[0][:, -1]
            
        # Compute penalty rewards if objects are in invalid ranges
        # penalty_rewards = (-1.0 - (metas.sum(dim=-1) / metas.size(-1))) if metas is not None else -1.0
        penalty_rewards = -1.0
    
        # Extract observation components
        object_pos, object_orient = next_observations[:, :3], next_observations[:, 3:6]
        object_quat = math_utils.quat_from_euler_xyz(object_orient[:, 0], object_orient[:, 1], object_orient[:, 2])
        hand_ee_pos, hand_ee_quat = next_observations[:, 6:9], math_utils.quat_from_euler_xyz(*next_observations[:, 9:12].T)
        finger_positions = next_observations[:, 12:27].reshape(-1, 5, 3)
                
        # Compute extrinsic rewards based on stage
        if stage_id == 0:  # Reach object
            grasp_range = self.grasp_range.to(encoded_goals.device)
            object_lengths = self.object_lengths.to(encoded_goals.device)
            # fingers cannot be too close to each other
            finger_distances = (
                (finger_positions.unsqueeze(1) - finger_positions.unsqueeze(2)).norm(dim=-1, p=2)
            ).permute(1, 2, 0)[torch.triu_indices(5, 5, offset=1).tolist()].permute(1, 0)
            not_sticky_scores = ((finger_distances[:, :4] / 0.025).clamp(0.0, 1.0).square() - 1.0).mean(dim=-1) \
                                + ((finger_distances[:, 4:] / 0.01).clamp(0.0, 1.0).square() - 1.0).mean(dim=-1)
            # fingers should reach the object
            delta_obj2fingers = torch.stack([
                math_utils.subtract_frame_transforms(q01=object_quat, t01=object_pos, t02=finger_pos)[0]
                for finger_pos in finger_positions.permute(1, 0, 2)
            ], dim=1).clamp(-grasp_range, grasp_range) / object_lengths
            reach_scores = 1.0 - ((delta_obj2fingers.norm(p=25, dim=-1).clamp(0.5, None) - 0.5) / ((grasp_range / object_lengths).norm(p=25) - 0.5)).sqrt().mean(dim=-1)
            # fingers should not be too near from the hand
            delta_hand2fingers = torch.stack([
                math_utils.subtract_frame_transforms(q01=hand_ee_quat, t01=hand_ee_pos, t02=finger_pos)[0]
                for finger_pos in finger_positions.permute(1, 0, 2)
            ], dim=1).clamp(-0.7 * object_lengths, 0.7 * object_lengths) / (0.7 * object_lengths)
            not_close_scores = delta_hand2fingers.norm(p=25, dim=-1).mean(dim=-1) - 1.0
            # combine scores to get intrinsic rewards
            delta_hand2obj = math_utils.subtract_frame_transforms(q01=hand_ee_quat, t01=hand_ee_pos, t02=object_pos)[0]
            combined_scores = torch.where(
                condition=(delta_hand2obj.abs() <= grasp_range).all(dim=-1),
                input=reach_scores, other=not_close_scores
            ) + not_sticky_scores
            # determine terminals and task rewards
            delta_goal = math_utils.subtract_frame_transforms(q01=hand_ee_quat, t01=hand_ee_pos, t02=encoded_goals[:, :3])[0]
            terminals = (delta_goal.abs() <= object_lengths / 2).all(dim=-1).float()
            # terminals = ((hand_ee_pos - encoded_goals[:, :3]).norm(dim=-1) <= 0.05).float()
            task_rewards = terminals + 0.5 * combined_scores
            # + combined_scores
        elif stage_id == 1:  # Align orientation (dense reward)
            # Expect encoded_goals shape (..., 10) with quaternion target at indices 6:10.
            # If shorter (<=6), orientation stage cannot be used; fallback to zero reward.
            if encoded_goals.shape[1] < 10:
                # Fallback: no orientation goal provided
                terminals = torch.zeros_like(penalty_rewards)
                task_rewards = terminals
            else:
                target_quat = encoded_goals[:, 6:10]
                # Ensure quaternions are unit (avoid NaNs)
                object_quat_n = object_quat / object_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                target_quat_n = target_quat / target_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                # Quaternion alignment error: angle between orientations
                # dot = cos(theta/2); angle = 2 * arccos(|dot|)
                dots = (object_quat_n * target_quat_n).sum(dim=-1).abs().clamp(0.0, 1.0)
                angles = 2.0 * torch.arccos(dots)  # in [0, pi]
                # Dense reward: higher when angle small. Normalize by pi.
                orient_dense = 1.0 - (angles / math_utils.PI)
                # Terminal when orientation within threshold (e.g., 10 degrees ~ 0.1745 rad)
                terminals = (angles <= 0.1745).float()
                task_rewards = self.orient_reward_scale * orient_dense + terminals
        elif stage_id == 2:  # Lift to goal (position)
            distances = (object_pos - encoded_goals[:, 3:6]).norm(dim=-1)
            terminals = (distances <= 0.03).float()
            task_rewards = terminals
        else:
            raise ValueError(f"Invalid stage_id: {stage_id}")
        
        # Compute final rewards
        rewards = self.scale_factor * (task_rewards + penalty_rewards)
        
        return rewards, terminals
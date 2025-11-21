from typing import Optional, Tuple
import torch

import isaaclab.utils.math as math_utils

###
##### REWARD FUNCTION FOR SINGLE-TASK SETUP
###

class FrankaCudeLiftReward:
    """Custom reward function for Franka-Shadow cube lifting task."""
    
    def __init__(self, scale_factor: float = 1.0):
        self.scale_factor = scale_factor
    
    def __call__(
        self, 
        next_observations: torch.Tensor, 
        goals: torch.Tensor, 
        metas: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rewards for the lift task.
        
        Args:
            next_observations: Current and next observations
            encoded_goals: Target encoded goals
            stage_id: Current stage (0=reach, 1=lift)
            metas: Meta information for penalties
        """
        # Pre-process
        if isinstance(next_observations, (list, tuple)):
            next_observations = next_observations[0][:, -1]
            
        # Compute penalty rewards if objects are in invalid ranges
        # penalty_rewards = (-1.0 - (metas.sum(dim=-1) / metas.size(-1))) if metas is not None else -1.0
        penalty_rewards = -1.0
    
        # Extract observation components
        object_pos = next_observations[:, :3]
        distances = (object_pos - goals).norm(dim=-1)
        terminals = (distances <= 0.03).float()
        rewards = self.scale_factor * (terminals + penalty_rewards)
        
        return rewards, terminals


###
##### REWARD FUNCTION FOR MULTI-TASK SETUP
###


class FrankaCudeMultiTaskLiftReward:
    """Custom reward function for Franka-Shadow cube lifting multi-task setup."""
    
    def __init__(self, scale_factor: float = 1.0):
        self.scale_factor = scale_factor
        self.object_lengths = torch.tensor([0.06, 0.06, 0.06])
        self.grasp_range = torch.tensor([0.12, 0.12, 0.09])
    
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
            stage_id: Current stage (0=reach, 1=lift)
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
        elif stage_id == 1:  # Lift to goal
            distances = (object_pos - encoded_goals[:, 3:6]).norm(dim=-1)
            terminals = (distances <= 0.03).float()
            task_rewards = terminals
        else:
            raise ValueError(f"Invalid stage_id: {stage_id}")
        
        # Compute final rewards
        rewards = self.scale_factor * (task_rewards + penalty_rewards)
        
        return rewards, terminals
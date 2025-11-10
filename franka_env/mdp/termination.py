from mdp.common import check_time_out, ManagerBasedRLEnv
from mdp.observation import get_command, get_object_position

# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm

###
##### TERMINAL PART
###

def check_goal_achieved(env: ManagerBasedRLEnv, dis_thresh: float):
    target_pos = get_command(env)
    object_pos = get_object_position(env)
    return ((target_pos - object_pos).norm(p=2, dim=-1) <= dis_thresh)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=check_time_out, time_out=True)

    goal_achieved = DoneTerm(func=check_goal_achieved, params={'dis_thresh': 0.03})
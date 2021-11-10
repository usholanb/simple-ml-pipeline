import torch
from modules.containers.di_containers import TrainerContainer
from modules.transformers.base_transformers.base_transformer import BaseTransformer
from utils.registry import registry


def one_hot_encode(inds, N, device=TrainerContainer.device):
    dims = [inds.size(i) for i in range(len(inds.size()))]
    inds = inds.unsqueeze(-1).long()
    dims.append(N)
    ret = (torch.zeros(dims)).to(device)
    ret.scatter_(-1, inds, 1)
    return ret


def to_goals_one_hot(original_goal, ohe_dim):
    return one_hot_encode(original_goal[:, :].data, ohe_dim)


@registry.register_transformer('goals_ohe')
class GoalsOHE(BaseTransformer):
    def apply(self, data):
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, \
            obs_goals, pred_goals_gt, seq_start_end = data
        goals_ohe = to_goals_one_hot(
            obs_goals, self.configs.get('special_inputs').get('g_dim')
        ).to(self.device)
        return obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, \
            goals_ohe, pred_goals_gt, seq_start_end


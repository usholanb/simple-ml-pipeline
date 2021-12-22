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
class GoalsOHETransformer(BaseTransformer):
    def apply(self, all_data):
        batch = all_data['batch']

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, \
            obs_goals, pred_goals_gt, seq_start_end = batch

        obs_goals_ohe = to_goals_one_hot(
            obs_goals, self.configs.get('special_inputs').get('g_dim')
        ).to(self.device)

        pred_goals_gt_ohe = to_goals_one_hot(
            pred_goals_gt, self.configs.get('special_inputs').get('g_dim')
        ).to(self.device)

        all_data['transformed_batch'] = obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, \
            obs_goals_ohe, pred_goals_gt_ohe, seq_start_end

        return all_data
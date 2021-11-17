"""
adj matrix for current batch
"""
import torch
import numpy as np
from modules.containers.di_containers import TrainerContainer
from modules.transformers.base_transformers.base_transformer import BaseTransformer
from utils.registry import registry
from scipy.spatial import distance_matrix


def permute2en(v, ndim_st=1):
    """
    Permute last ndim_en of tensor v to the first
    :type v: torch.Tensor
    :type ndim_st: int
    :rtype: torch.Tensor
    """
    nd = v.ndimension()
    return v.permute([*range(ndim_st, nd)] + [*range(ndim_st)])


def permute2st(v, ndim_en=1):
    """
    Permute last ndim_en of tensor v to the first
    :type v: torch.Tensor
    :type ndim_en: int
    :rtype: torch.Tensor
    """
    nd = v.ndimension()
    return v.permute([*range(-ndim_en, 0)] + [*range(nd - ndim_en)])


def compute_adjs(t, seq_start_end):
    adj_out = []
    for _, (start, end) in enumerate(seq_start_end):
        mat = []
        for t in range(0, t.obs_len + t.pred_len):
            interval = end - start
            mat.append(torch.from_numpy(np.ones((interval, interval))))
        adj_out.append(torch.stack(mat, 0))
    return block_diag_irregular(adj_out)


def block_diag_irregular(matrices):
    if len(matrices[0].shape) == 3:
        l, b, h = matrices[0].shape
        total_num = len(matrices)

        m_lst = [matrices[i].reshape(l, b*h) for i in range(len(matrices))]
        v = torch.block_diag(*m_lst)
        v = v.reshape([l * total_num, b * total_num, h])
    else:
        v = torch.block_diag(*matrices) # v_2
    return permute2en(v, 2)


def compute_adjs_knnsim(special_inputs, seq_start_end, obs_traj, pred_traj_gt):
    adj_out = []
    for _, (start, end) in enumerate(seq_start_end):
        obs_and_pred_traj = torch.cat((obs_traj, pred_traj_gt))
        knn_t = []
        for t in range(0, special_inputs.obs_len + special_inputs.pred_len):
            dists = distance_matrix(np.asarray(obs_and_pred_traj[t, start:end, :]),
                                    np.asarray(obs_and_pred_traj[t, start:end, :]))
            knn = np.argsort(dists, axis=1)[:, 0: min(special_inputs.top_k_neigh, dists.shape[0])]
            final_dists = []
            for i in range(dists.shape[0]):
                knni = np.zeros((dists.shape[1],))
                knni[knn[i]] = 1
                final_dists.append(knni)
            final_dists = np.stack(final_dists)
            knn_t.append(torch.from_numpy(final_dists))
        adj_out.append(torch.stack(knn_t, 0))
    return block_diag_irregular(adj_out)


def compute_adjs_distsim(t_config, seq_start_end, obs_traj, pred_traj_gt):
    adj_out = []
    for _, (start, end) in enumerate(seq_start_end):
        obs_and_pred_traj = torch.cat((obs_traj, pred_traj_gt))
        sim_t = []
        for t in range(0, t_config['obs_len'] + t_config['pred_len']):
            dists = distance_matrix(np.asarray(obs_and_pred_traj[t, start:end, :]),
                                    np.asarray(obs_and_pred_traj[t, start:end, :]))
            #sum_dist = np.sum(dists)
            #dists = np.divide(dists, sum_dist)
            sim = np.exp(-dists / t_config['sigma'])
            sim_t.append(torch.from_numpy(sim))
        adj_out.append(torch.stack(sim_t, 0))
    return block_diag_irregular(adj_out)


def compute_adjs_distsim_v2(si, seq_start_end, obs_traj, pred_traj_gt):
    num_agents = si.get('n_max_agents')
    sigma = torch.tensor(si.get('sigma'), device=torch.device('cuda'))
    obs_and_pred_traj = torch.cat((obs_traj, pred_traj_gt))
    seq_len, num_traj, xy = obs_and_pred_traj.shape
    obs_and_pred_traj_ts = torch.reshape(obs_and_pred_traj, (seq_len, num_traj // num_agents, num_agents, xy))
    timestep, batch, _, _ = obs_and_pred_traj_ts.shape
    obs_and_pred_traj_batch = obs_and_pred_traj_ts.permute(1, 0, 2, 3).reshape((batch * timestep, num_agents, xy))

    dists = torch.cdist(obs_and_pred_traj_batch, obs_and_pred_traj_batch)
    batch_dist, _, _ = dists.shape
    sim = torch.exp(-dists / sigma).reshape(batch, timestep, num_agents, num_agents)
    sim = torch.permute(sim, (0, 2, 3, 1))

    return block_diag_irregular(torch.unbind(sim))


@registry.register_transformer('adj')
class AdjTransformer(BaseTransformer):

    def __init__(self, configs):
        super(AdjTransformer, self).__init__(configs)
        self.adj_train = []
        self.adj_valid = []
        self.adj_test = []

    def apply(self, all_data):

        batch = all_data['transformed_batch']

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, \
        obs_goals_ohe, pred_goals_gt_ohe, seq_start_end = batch

        if all_data['split'] == 'train':
            adj = self.adj_train
        elif all_data['split'] == 'valid':
            adj = self.adj_valid
        elif all_data['split'] == 'test':
            adj = self.adj_test

        batch_i = all_data['batch_i']
        if len(adj) > batch_i:
            adj_out = adj[batch_i]
        else:
            si = self.configs.get('special_inputs')
            seq_len = len(obs_traj) + len(pred_traj_gt)
            assert seq_len == si.get('obs_len') + si.get('pred_len')
            if si.get('adjacency_type') == 0:
                adj_out = compute_adjs(self.configs.get('special_inputs'), seq_start_end).to(self.device)
            elif si.get('adjacency_type') == 1:
                adj_out = compute_adjs_distsim_v2(self.configs.get('special_inputs'), seq_start_end, obs_traj,
                                                  pred_traj_gt)
            elif si.get('adjacency_type') == 2:
                adj_out = compute_adjs_knnsim(self.configs.get('special_inputs'), seq_start_end,
                                              obs_traj.detach().cpu(),
                                              pred_traj_gt.detach().cpu()).to(self.device)
            adj.append(adj_out.to(self.device).float())

        all_data['transformed_batch'] = obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, \
            obs_goals_ohe, pred_goals_gt_ohe, seq_start_end, adj_out




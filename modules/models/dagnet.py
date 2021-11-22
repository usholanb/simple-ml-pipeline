import torch
import math
from torch.nn import functional as F
from modules.containers.di_containers import TrainerContainer
from modules.models.base_models.base_torch_model import BaseTorchModel
from modules.transformers.adj_transformer import block_diag_irregular
from utils.registry import registry
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.nn as nn
from scipy.spatial import distance_matrix


@registry.register_model('dagnet')
class DAGNet (BaseTorchModel):
    def __init__(self, configs):
        super(DAGNet, self).__init__(configs)
        self.d_dim = self.n_max_agents * 2
        self.init_local_variables()

        # goal generator
        self.dec_goal = nn.Sequential(
            nn.Linear(self.d_dim + self.g_dim + self.rnn_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.g_dim),
            nn.Softmax(dim=-1)
        )

        # goals graph
        if self.adjacency_type == 2 and self.top_k_neigh is None:
            raise Exception('Using KNN-similarity but top_k_neigh is not specified')

        if self.graph_model == 'gcn':
            self.graph_goals = GCN(self.g_dim, self.graph_hid, self.g_dim)
        elif self.graph_model == 'gat':
            assert self.n_heads is not None
            assert self.alpha is not None
            self.graph_goals = GAT(self.g_dim, self.graph_hid, self.g_dim, self.alpha, self.n_heads)

        # hiddens graph
        if self.adjacency_type == 2 and self.top_k_neigh is None:
            raise Exception('Using KNN-similarity but top_k_neigh is not specified')
        if self.graph_model == 'gcn':
            self.graph_hiddens = GCN(self.rnn_dim, self.graph_hid, self.rnn_dim)
        elif self.graph_model == 'gat':
            assert self.n_heads is not None
            assert self.alpha is not None
            self.graph_hiddens = GAT(self.rnn_dim, self.graph_hid, self.rnn_dim, self.alpha, self.n_heads)

        # interpolating original goals with refined goals from the first graph
        self.lg_goals = nn.Sequential(
            nn.Linear(self.g_dim + self.g_dim, self.g_dim),
            nn.Softmax(dim=-1)
        )

        # interpolating original hiddens with refined hiddens from the second graph
        self.lg_hiddens = nn.Linear(self.rnn_dim + self.rnn_dim, self.rnn_dim)

        # feature extractors
        self.phi_x = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU()
        )

        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU()
        )

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(self.h_dim + self.g_dim + self.rnn_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU()
        )
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)
        self.enc_logvar = nn.Linear(self.h_dim, self.z_dim)

        # prior
        self.prior = nn.Sequential(
            nn.Linear(self.g_dim + self.rnn_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU()
        )
        self.prior_mean = nn.Linear(self.h_dim, self.z_dim)
        self.prior_logvar = nn.Linear(self.h_dim, self.z_dim)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(self.d_dim + self.g_dim + self.h_dim + self.rnn_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU()
        )
        self.dec_mean = nn.Linear(self.h_dim, self.x_dim)
        self.dec_logvar = nn.Linear(self.h_dim, self.x_dim)

        # recurrence
        self.rnn = nn.GRU(self.h_dim + self.h_dim, self.rnn_dim, self.n_layers)

    def init_local_variables(self):
        self.total_traj = None
        self.ade_outer = None
        self.fde_outer = None
        self.ade = None
        self.fde = None
        self.loss = None
        self.kld_loss = None
        self.nll_loss = None
        self.cross_entropy_loss = None
        self.euclidean_loss = None

    def _reparameterize(self, mean, log_var):
        logvar = torch.exp(log_var * 0.5).to(self.device)
        eps = torch.rand_like(logvar).to(self.device)
        return eps.mul(logvar).add(mean)

    def _kld(self, mean_enc, logvar_enc, mean_prior, logvar_prior):
        x1 = torch.sum((logvar_prior - logvar_enc), dim=1)
        x2 = torch.sum(torch.exp(logvar_enc - logvar_prior), dim=1)
        x3 = torch.sum((mean_enc - mean_prior).pow(2) / (torch.exp(logvar_prior)), dim=1)
        kld_element = x1 - mean_enc.size(1) + x2 + x3
        return torch.mean(0.5 * kld_element)

    def _nll_gauss(self, mean, logvar, x):
        x1 = torch.sum(((x - mean).pow(2)) / torch.exp(logvar), dim=1)
        x2 = x.size(1) * np.log(2 * np.pi)
        x3 = torch.sum(logvar, dim=1)
        nll = torch.mean(0.5 * (x1 + x2 + x3))
        return nll

    def forward(self, all_data):
        traj, traj_rel, goals_ohe, seq_start_end, adj_out = all_data['forward_data']
        timesteps, batch, features = traj.shape

        d = torch.zeros(timesteps, batch, features*self.n_max_agents).to(self.device)
        h = torch.zeros(self.n_layers, batch, self.rnn_dim).to(self.device)

        # an agent has to know all the xy abs positions of all the other agents in its sequence (for every timestep)
        for idx, (start,end) in enumerate(seq_start_end):
            n_agents = (end-start).item()
            d[:, start:end, :n_agents*2] = traj[:, start:end, :].reshape(timesteps, -1).unsqueeze(1).repeat(1,n_agents,1)

        KLD = torch.zeros(1).to(self.device)
        NLL = torch.zeros(1).to(self.device)
        cross_entropy = torch.zeros(1).to(self.device)

        for timestep in range(1, timesteps):
            x_t = traj_rel[timestep]
            d_t = d[timestep]
            g_t = goals_ohe[timestep]   # ground truth goal
            # refined goal must resemble real goal g_t
            dec_goal_t = self.dec_goal(torch.cat([d_t, h[-1], goals_ohe[timestep-1]], 1))
            g_graph = self.graph_goals(dec_goal_t, adj_out[timestep])  # graph refinement
            g_combined = self.lg_goals(torch.cat((dec_goal_t, g_graph), dim=-1))     # combination
            cross_entropy -= torch.sum(g_t * g_combined)

            # input feature extraction and encoding
            phi_x_t = self.phi_x(x_t)
            enc_t = self.enc(torch.cat([phi_x_t, g_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_logvar_t = self.enc_logvar(enc_t)

            # prior
            prior_t = self.prior(torch.cat([g_t, h[-1]], 1))
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)

            # sampling from latent
            z_t = self._reparameterize(enc_mean_t, enc_logvar_t)

            # z_t feature extraction and decoding
            phi_z_t = self.phi_z(z_t)
            dec_t = self.dec(torch.cat([d_t, g_t, phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_logvar_t = self.dec_logvar(dec_t)

            # agent vrnn recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            KLD += self._kld(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
            NLL += self._nll_gauss(dec_mean_t, dec_logvar_t, x_t)

            # hidden states refinement with graph
            h_graph = self.graph_hiddens(h[-1].clone(), adj_out[timestep])  # graph refinement
            h[-1] = self.lg_hiddens(torch.cat((h_graph, h[-1]), dim=-1)).unsqueeze(0)  # combination

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, \
        obs_goals_ohe, pred_goals_gt_ohe, seq_start_end, adj_out = all_data['transformed_batch']
        seq_len = len(obs_traj) + len(pred_traj_gt)
        samples_rel = self.predict_proba(self.pred_len, h, obs_traj[-1],
                                         obs_goals_ohe[-1], seq_start_end)

        samples = relative_to_abs(samples_rel, obs_traj[-1])
        batch_traj = obs_traj.shape[1]  # num_seqs
        ade_traj = average_displacement_error(samples, pred_traj_gt)

        euclidean_batch_loss = ade_traj / (batch_traj * seq_len)
        all_data['outputs'] = {
            'kld': KLD,
            'nll': NLL,
            'cross_entropy': cross_entropy,
            'h': h,
            'euclidean_loss': euclidean_batch_loss,
        }

    def predict_proba(self, samples_seq_len, h, x_abs_start, g_start, seq_start_end):
        _, batch_size, _ = h.shape

        g_t = g_start   # at start, the previous goal is the last goal from GT observation
        x_t_abs = x_abs_start # at start, the curr abs pos of the agents come from the last abs pos from GT observations

        samples = torch.zeros(samples_seq_len, batch_size, self.x_dim).to(self.device)
        d = torch.zeros(samples_seq_len, batch_size, self.n_max_agents * self.x_dim).to(self.device)
        displacements = torch.zeros(samples_seq_len, batch_size, self.n_max_agents * 2).to(self.device)

        # at start, the disposition of the agents is composed by the last abs positions from GT obs
        for idx, (start,end) in enumerate(seq_start_end):
            n_agents = (end-start).item()
            d[0, start:end, :n_agents*2] = x_abs_start[start:end].reshape(-1).repeat(n_agents, 1)

        with torch.no_grad():
            for timestep in range(samples_seq_len):
                d_t = d[timestep]

                if self.adjacency_type == 0:
                    adj_pred = adjs_fully_connected_pred(seq_start_end)
                elif self.adjacency_type == 1:
                    adj_pred = adjs_distance_sim_pred(self.sigma, seq_start_end, x_t_abs.detach().cpu())
                elif self.adjacency_type == 2:
                    adj_pred = adjs_knn_sim_pred(self.top_k_neigh, seq_start_end, x_t_abs.detach().cpu())
                adj_pred = adj_pred.to(self.device).float()

                # sampling agents' goals + graph refinement step
                dec_g = self.dec_goal(torch.cat([d_t, h[-1], g_t], 1))
                g_graph = self.graph_goals(dec_g, adj_pred)
                g_combined = self.lg_goals(torch.cat((dec_g, g_graph), dim=-1))
                g_t = sample_multinomial(torch.exp(g_combined))     # final predicted goal at current t

                # prior
                prior_t = self.prior(torch.cat([g_t, h[-1]], 1))
                prior_mean_t = self.prior_mean(prior_t)
                prior_logvar_t = self.prior_logvar(prior_t)

                # sampling from latent
                z_t = self._reparameterize(prior_mean_t, prior_logvar_t)

                # z_t feature extraction and decoding
                phi_z_t = self.phi_z(z_t)
                dec_t = self.dec(torch.cat([d_t, g_t, phi_z_t, h[-1]], 1))
                dec_mean_t = self.dec_mean(dec_t)
                dec_logvar_t = self.dec_logvar(dec_t)
                samples[timestep] = dec_mean_t

                # feature extraction for reconstructed samples
                phi_x_t = self.phi_x(dec_mean_t)

                # vrnn recurrence
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

                # graph refinement for agents' hiddens
                if self.adjacency_type == 0:
                    adj_pred = adjs_fully_connected_pred(seq_start_end)
                elif self.adjacency_type == 1:
                    adj_pred = adjs_distance_sim_pred(self.sigma, seq_start_end, x_t_abs.detach().cpu())
                elif self.adjacency_type == 2:
                    adj_pred = adjs_knn_sim_pred(self.top_k_neigh, seq_start_end, x_t_abs.detach().cpu())
                adj_pred = adj_pred.to(self.device).float()
                h_graph = self.graph_hiddens(h[-1].clone(), adj_pred)
                h[-1] = self.lg_hiddens(torch.cat((h_graph, h[-1]), dim=-1)).unsqueeze(0)

                # new abs pos
                x_t_abs = x_t_abs + dec_mean_t

                # disposition at t+1 is the current disposition d_t + the predicted displacements (dec_mean_t)
                if timestep != (samples_seq_len - 1):
                    for idx, (start, end) in enumerate(seq_start_end):
                        n_agents = (end - start).item()
                        displacements[timestep, start:end, :n_agents*2] = dec_mean_t[start:end].reshape(-1).repeat(n_agents, 1)
                    d[timestep + 1, :, :] = d_t[:, :] + displacements[timestep]

        return samples

    def before_epoch_train(self):
        self.loss = 0
        self.kld_loss = 0
        self.nll_loss = 0
        self.cross_entropy_loss = 0
        self.euclidean_loss = 0

    def before_iteration_valid(self, all_data):
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, \
        obs_goals_ohe, pred_goals_gt_ohe, seq_start_end, adj_out = all_data['transformed_batch']
        all_data['forward_data'] = obs_traj, obs_traj_rel, obs_goals_ohe, seq_start_end, adj_out

    def before_epoch_eval(self):
        self.before_epoch_train()
        self.total_traj = 0
        self.ade_outer = []
        self.fde_outer = []
        self.ade = None
        self.fde = None

    def before_iteration_train(self, all_data):
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, \
        obs_goals_ohe, pred_goals_gt_ohe, seq_start_end, adj_out = all_data['transformed_batch']
        all_traj = torch.cat((obs_traj, pred_traj_gt), dim=0)
        all_traj_rel = torch.cat((obs_traj_rel, pred_traj_rel_gt), dim=0)
        all_goals_ohe = torch.cat((obs_goals_ohe, pred_goals_gt_ohe), dim=0)
        all_data['forward_data'] = all_traj, all_traj_rel, all_goals_ohe, seq_start_end, adj_out

    def after_epoch_train(self, split_name, loader):
        return {
            f'{split_name}_avg_loss': self.loss / len(loader.dataset),
            f'{split_name}_avg_kld_loss': self.kld_loss / len(loader.dataset),
            f'{split_name}_avg_nll_loss': self.nll_loss / len(loader.dataset),
            f'{split_name}_avg_cross_entropy_loss': self.cross_entropy_loss / len(loader.dataset),
            f'{split_name}_mean_euclidean_loss': self.euclidean_loss / len(loader.dataset),
        }

    def after_epoch_valid(self, split_name, loader):
        losses = self.after_epoch_train(split_name, loader)
        ade = sum(self.ade_outer) / (self.total_traj * self.pred_len)
        fde = sum(self.fde_outer) / self.total_traj
        losses.update({
            f'{split_name}_total_traj': self.total_traj,
            f'{split_name}_ade': ade,
            f'{split_name}_fde': fde,
        })
        self.init_local_variables()
        return losses

    def end_iteration_train(self, all_data):
        loss_outputs = all_data['loss_outputs']
        self.loss += loss_outputs['loss'].item()
        self.kld_loss += loss_outputs['kld_loss']
        self.nll_loss += loss_outputs['nll_loss']
        self.cross_entropy_loss += loss_outputs['cross_entropy_loss']
        self.euclidean_loss += loss_outputs['euclidean_loss']

    def end_iteration_valid(self, all_data):
        data, forward_data, outputs = all_data['batch'], all_data['forward_data'], all_data['outputs'],
        obs_traj, obs_traj_rel, obs_goals_ohe, seq_start_end, adj_out = forward_data
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, \
        obs_goals, pred_goals_gt, seq_start_end = data
        ade, fde = [], []
        self.total_traj += obs_traj.shape[1]

        for _ in range(self.configs.get('trainer').get('num_samples')):
            samples_rel = self.predict_proba(self.pred_len, outputs['h'], obs_traj[-1], obs_goals_ohe[-1], seq_start_end)
            samples = relative_to_abs(samples_rel, obs_traj[-1])

            ade.append(average_displacement_error(samples, pred_traj_gt, mode='raw'))
            fde.append(final_displacement_error(samples[-1, :, :], pred_traj_gt[-1, :, :], mode='raw'))

        ade_sum = evaluate_helper(ade, seq_start_end).item()
        fde_sum = evaluate_helper(fde, seq_start_end).item()

        self.ade_outer.append(ade_sum)
        self.fde_outer.append(fde_sum)


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


class GCN(nn.Module):
    def __init__(self, nin, nhid, nout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nin, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.bn2 = nn.BatchNorm1d(nout)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.bn1(x)
        x = self.gc2(x, adj)
        x = self.bn2(x)
        return torch.tanh(x)


# fully connected
def adjs_fully_connected(seq_len, seq_start_end):
    adj_out = []

    for _, (start, end) in enumerate(seq_start_end):
        mat = []
        for t in range(0, seq_len):
            interval = end - start
            mat.append(torch.from_numpy(np.ones((interval, interval))))

        adj_out.append(torch.stack(mat, 0))

    return block_diag_irregular(adj_out)


def adjs_fully_connected_pred (seq_start_end):
    adj_out = []

    for _, (start, end) in enumerate(seq_start_end):
        interval = end - start
        adj_out.append(torch.from_numpy(np.ones((interval, interval))))

    return block_diag_irregular(adj_out)


# adjacency with distance similarity
def adjs_distance_sim(seq_len, sigma, seq_start_end, traj):
    adj_out = []

    for _, (start, end) in enumerate(seq_start_end):
        sim_t = []

        for t in range(0, seq_len):
            dists = distance_matrix(np.asarray(traj[t, start:end, :]),
                                    np.asarray(traj[t, start:end, :]))
            sim = np.exp(-dists / sigma)
            sim_t.append(torch.from_numpy(sim))

        adj_out.append(torch.stack(sim_t, 0))

    return block_diag_irregular(adj_out)


def adjs_distance_sim_pred(sigma, seq_start_end, pred_traj):
    adj_out = []

    for _, (start, end) in enumerate(seq_start_end):
        dists = distance_matrix(np.asarray(pred_traj[start:end, :]),
                                np.asarray(pred_traj[start:end, :]))
        sim = np.exp(-dists / sigma)
        adj_out.append(torch.from_numpy(sim))

    return block_diag_irregular(adj_out)


# adjacency with KNN similarity
def adjs_knn_sim(seq_len, top_k_neigh, seq_start_end, traj):
    adj_out = []

    for _, (start, end) in enumerate(seq_start_end):
        knn_t = []
        for t in range(0, seq_len):
            dists = distance_matrix(np.asarray(traj[t, start:end, :]),
                                    np.asarray(traj[t, start:end, :]))
            knn = np.argsort(dists, axis=1)[:, 0: min(top_k_neigh, dists.shape[0])]

            final_dists = []
            for i in range(dists.shape[0]):
                knni = np.zeros((dists.shape[1],))
                knni[knn[i]] = 1
                final_dists.append(knni)

            final_dists = np.stack(final_dists)
            knn_t.append(torch.from_numpy(final_dists))

        adj_out.append(torch.stack(knn_t, 0))

    return block_diag_irregular(adj_out)


def adjs_knn_sim_pred(top_k_neigh, seq_start_end, pred_traj):
    adj_out = []

    for _, (start, end) in enumerate(seq_start_end):
        dists = distance_matrix(np.asarray(pred_traj[start:end, :]),
                                np.asarray(pred_traj[start:end, :]))
        knn = np.argsort(dists, axis=1)[:, 0: min(top_k_neigh, dists.shape[0])]

        final_dists = []
        for i in range(dists.shape[0]):
            knni = np.zeros((dists.shape[1],))
            knni[knn[i]] = 1
            final_dists.append(knni)

        final_dists = np.stack(final_dists)
        adj_out.append(torch.from_numpy(final_dists))

    return block_diag_irregular(adj_out)


###################################### ORIGINAL ADJ MATRICES ######################################


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


def compute_adjs_knnsim_pred(top_k_neigh, seq_start_end, pred_traj):
    adj_out = []
    for _, (start, end) in enumerate(seq_start_end):
        dists = distance_matrix(np.asarray(pred_traj[start:end, :]),
                                np.asarray(pred_traj[start:end, :]))
        knn = np.argsort(dists, axis=1)[:, 0: min(top_k_neigh, dists.shape[0])]
        final_dists = []
        for i in range(dists.shape[0]):
            knni = np.zeros((dists.shape[1],))
            knni[knn[i]] = 1
            final_dists.append(knni)
        final_dists = np.stack(final_dists)
        adj_out.append(torch.from_numpy(final_dists))
    return block_diag_irregular(adj_out)


def compute_adjs_distsim_pred(sigma, seq_start_end, pred_traj):
    adj_out = []
    for _, (start, end) in enumerate(seq_start_end):
        dists = distance_matrix(np.asarray(pred_traj[start:end, :]),
                                np.asarray(pred_traj[start:end, :]))
        sim = np.exp(-dists / sigma)
        adj_out.append(torch.from_numpy(sim))
    return block_diag_irregular(adj_out)


######################### MISCELLANEOUS ############################
def relative_to_abs(traj_rel, start_pos):
    """
    Inputs:
    - rel_traj: tensor of shape (seq_len, batch, 2), trajectory composed by displacements
    - start_pos: tensor of shape (batch, 2), initial position
    Outputs:
    - input tensor (seq_len, batch, 2) filled with absolute coords
    """
    rel_traj = traj_rel.permute(1, 0, 2)        # (seq_len, batch, 2) -> (batch, seq_len, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos

    return abs_traj.permute(1, 0, 2)


def plot_traj(observed, predicted_gt, predicted, seq_start_end, writer, epch):
    """
    Inputs:
    - predicted: tensor (pred_len, batch, 2) with predicted trajectories
    - observed: tensor (obs_len, batch, 2) with observed trajectories
    - predicted_gt: tensor (pred_len, batch, 2) with predicted trajectories ground truth
    - seq_start_end: tensor (num_seq, 2) with temporal sequences start and end
    - writer: Tensorboard writer
    - epch: current epoch
    """

    idx = random.randrange(0, len(seq_start_end))    # print only one (random) sequence in the batch
    (start, end) = seq_start_end[idx]
    pred = predicted[:, start:end, :]
    obs = observed[:, start:end, :]
    pred_gt = predicted_gt[:, start:end, :]

    fig = plt.figure()

    for idx in range(pred.shape[1]):
        # draw observed
        plt.plot(obs[:, idx, 0], obs[:, idx, 1], color='green')

        # draw connection between last observed point and first predicted gt point
        x1, x2 = obs[-1, idx, 0], pred_gt[0, idx, 0]
        y1, y2 = obs[-1, idx, 1], pred_gt[0, idx, 1]
        plt.plot([x1, x2], [y1, y2], color='blue')

        # draw predicted gt
        plt.plot(pred_gt[:, idx, 0], pred_gt[:, idx, 1], color='blue')

        # draw connection between last observed point and first predicted point
        x1, x2 = obs[-1, idx, 0], pred[0, idx, 0]
        y1, y2 = obs[-1, idx, 1], pred[0, idx, 1]
        plt.plot([x1, x2], [y1, y2], color='red')

        # draw predicted
        plt.plot(pred[:, idx, 0], pred[:, idx, 1], color='red')

    writer.add_figure('Generations', fig, epch)
    fig.clf()


def std(vector, mean):
    sum = torch.zeros(mean.shape).to(device)
    for el in vector:
        sum += el - mean
    return torch.sqrt(torch.abs(sum) / len(vector))



######################### LOSSES ############################
def average_displacement_error(pred_traj, pred_traj_gt, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectories.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth predictions.
    Output:
    - error: total sum of displacement errors across all sequences inside the batch
    """

    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss ** 2
    loss = torch.sqrt(torch.sum(loss, dim=2))
    loss = torch.sum(loss, dim=1)

    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, mode='sum'):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted final positions.
    - pred_pos_gt: Tensor of shape (batch, 2). Ground truth final positions.
    Output:
    - error: total sum of fde for all the sequences inside the batch
    """

    loss = (pred_pos - pred_pos_gt) ** 2
    loss = torch.sqrt(torch.sum(loss, dim=1))

    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode='sum'):
    """
    :param pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    :param pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    :param mode: Can be one of sum, average or raw
    :return: l2 loss depending on mode
    """

    loss = (loss_mask.to(device).unsqueeze(dim=2) * (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)


def linear_velocity_acceleration(sequence, mode='mean', seconds_between_frames=0.2):
    seq_len, batch, features = sequence.shape

    velocity_x = torch.zeros((seq_len, batch)).to(device)
    velocity_y = torch.zeros((seq_len, batch)).to(device)
    for ped in range(batch):
        velocity_x[1:, ped] = (sequence[1:, ped, 0] - sequence[:-1, ped, 0]) / seconds_between_frames     # space / time
        velocity_y[1:, ped] = (sequence[1:, ped, 1] - sequence[:-1, ped, 1]) / seconds_between_frames     # space / time
    velocity = torch.sqrt(torch.pow(velocity_x, 2) + torch.pow(velocity_y, 2))

    acceleration_x = torch.zeros((seq_len, batch)).to(device)
    acceleration_y = torch.zeros((seq_len, batch)).to(device)
    acceleration = torch.zeros((seq_len, batch)).to(device)
    acceleration_x[2:, :] = (velocity_x[2:, :] - velocity_x[1:-1, :]) / seconds_between_frames
    acceleration_y[2:, :] = (velocity_y[2:, :] - velocity_y[1:-1, :]) / seconds_between_frames
    acceleration_comp = torch.sqrt(torch.pow(acceleration_x, 2) + torch.pow(acceleration_y, 2))
    acceleration[2:, :] = torch.abs((velocity[2:, :] - velocity[1:-1, :]) / seconds_between_frames)

    # Linear velocity means
    mean_velocity_x = velocity_x[1:].mean(dim=0)
    mean_velocity_y = velocity_y[1:].mean(dim=0)
    mean_velocity = velocity[1:].mean(dim=0)

    # Linear acceleration means
    mean_acceleration_x = acceleration_x[2:].mean(dim=0)
    mean_acceleration_y = acceleration_y[2:].mean(dim=0)
    mean_acceleration = acceleration[2:].mean(dim=0)

    # Linear velocity standard deviations
    std_velocity_x = std(velocity_x, mean_velocity_x)
    std_velocity_y = std(velocity_y, mean_velocity_y)
    sum_velocity = torch.sqrt(torch.pow(std_velocity_x, 2) + torch.pow(std_velocity_y, 2))
    std_velocity = torch.sqrt(torch.abs(sum_velocity) / len(sequence))

    # Linear acceleration standard deviations
    std_acceleration_x = std(mean_acceleration_x, mean_acceleration_x)
    std_acceleration_y = std(mean_acceleration_y, mean_acceleration_y)
    sum_acceleration = torch.sqrt(torch.pow(std_acceleration_x, 2) + torch.pow(std_acceleration_y, 2))
    std_acceleration = torch.sqrt(torch.abs(sum_acceleration) / len(sequence))

    if mode == 'raw':
        return mean_velocity, mean_acceleration, std_velocity_x, std_velocity_y
    elif mode == 'mean':
        # mode mean
        return mean_velocity.mean(), mean_acceleration.mean(), std_velocity.mean(), \
                std_acceleration.mean()
    else:
        raise(Exception("linear_velocity_acceleration(): wrong mode selected. Choices are ['raw', 'mean'].\n"))


######################### GOALS UTILITIES ############################
def one_hot_encode(inds, N):
    device = TrainerContainer.device
    dims = [inds.size(i) for i in range(len(inds.size()))]
    inds = inds.unsqueeze(-1).long()
    dims.append(N)
    ret = (torch.zeros(dims)).to(device)
    ret.scatter_(-1, inds, 1)
    return ret


def to_goals_one_hot(original_goal, ohe_dim):
    return one_hot_encode(original_goal[:, :].data, ohe_dim)


def divide_by_agents(data, n_agents):
    x = data[:, :, :2 * n_agents].clone()
    return x.view(x.size(0), x.size(1), n_agents, -1).transpose(1, 2)


def sample_multinomial(probs):
    """ Each element of probs tensor [shape = (batch, g_dim)] has 'g_dim' probabilities (one for each grid cell),
    i.e. is a row containing a probability distribution for the goal. We sample n (=batch) indices (one for each row)
    from these distributions, and covert it to a 1-hot encoding. """

    inds = torch.multinomial(probs, 1).data.long().squeeze()
    ret = one_hot_encode(inds, probs.size(-1))
    return ret


def get_goal(position, min_x, max_x, min_y, max_y, n_cells_x, n_cells_y):
    """Divides the scene rectangle into a grid of cells and finds the cell idx where the current position falls"""
    x = position[0]
    y = position[1]

    x_steps = np.linspace(min_x, max_x, num=n_cells_x+1)
    y_steps = np.linspace(min_y, max_y, num=n_cells_y+1)

    goal_x_idx = np.max(np.where(x_steps <= x)[0])
    goal_y_idx = np.max(np.where(y_steps <= y)[0])
    return (goal_x_idx * n_cells_y) + goal_y_idx


def compute_goals_fixed(tj, min_x, max_x, min_y, max_y, n_cells_x, n_cells_y, window=1):
    """Computes goals using the position at the end of each window."""
    timesteps = len(tj)
    goals = np.zeros(timesteps)
    for t in reversed(range(timesteps)):
        if (t + 1) % window == 0 or (t + 1) == timesteps:
            goals[t] = get_goal(tj[t], min_x, max_x, min_y, max_y, n_cells_x, n_cells_y)
        else:
            goals[t] = goals[t + 1]

    return goals


######################### MATRIX UTILITIES #############################
# FROM:  https://github.com/yulkang/pylabyk/blob/master/numpytorch.py
def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1).to(device)
    return (m2 * eye).reshape(siz0 + torch.Size(torch.tensor(siz1) * n))


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))


class GAT(nn.Module):
    """Dense version of GAT."""
    def __init__(self, nin, nhid, nout, alpha, nheads):
        super(GAT, self).__init__()

        self.attentions = [GraphAttentionLayer(nin, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nout, alpha=alpha, concat=False)
        self.bn1 = nn.BatchNorm1d(nout)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.out_att(x, adj)
        x = self.bn1(x)
        return torch.tanh(x)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)  # matrix multiplication of the matrices
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).to(TrainerContainer.device)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features)).to(TrainerContainer.device)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)
        # self.bias.data.fill_(0.01)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

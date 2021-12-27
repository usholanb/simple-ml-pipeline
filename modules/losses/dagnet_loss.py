from typing import Dict
import torch
import numpy as np
from modules.containers.di_containers import TrainerContainer
from modules.losses.base_losses.base_loss import BaseLoss
from utils.common import std
from utils.registry import registry


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
    device = TrainerContainer.device
    loss = (loss_mask.to(device).unsqueeze(dim=2) * (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)


def linear_velocity_acceleration(sequence, mode='mean', seconds_between_frames=0.2):
    seq_len, batch, features = sequence.shape
    device = TrainerContainer.device
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


@registry.register_loss('dagnet_loss')
class DagnetLoss(BaseLoss):
    def __init__(self, configs):

        self.configs = configs
        si = self.configs.get('special_inputs')
        warmup = si.get('warmup')
        wrmp_epochs = si.get('wrmp_epochs')
        num_epochs = self.configs.get('trainer').get('epochs')
        self.warmup = np.ones(num_epochs)
        self.warmup[:wrmp_epochs] = np.linspace(0, 1, num=wrmp_epochs)\
            if warmup else self.warmup[:wrmp_epochs]
        self.CE_weight = si.get('CE_weight')
        self._lambda = si.get('_lambda')

    def __call__(self, all_data: Dict) -> Dict:
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, \
        obs_goals_ohe, pred_goals_gt_ohe, seq_start_end, adj_out = all_data['transformed_batch']
        seq_len = len(obs_traj) + len(pred_traj_gt)
        batch_traj = obs_traj.shape[1]  # num_seqs
        ade_traj = average_displacement_error(all_data['samples'], pred_traj_gt)

        euclidean_batch_loss = ade_traj / (batch_traj * seq_len)
        all_data['outputs'] = {
            'kld': all_data.pop('KLD'),
            'nll': all_data.pop('NLL'),
            'cross_entropy': all_data.pop('cross_entropy'),
            'h': all_data.pop('h'),
            'euclidean_loss': euclidean_batch_loss,
        }

        outputs = all_data['outputs']
        kld, nll, cross_entropy, euclidean_loss = \
            outputs['kld'], outputs['nll'], outputs['cross_entropy'], outputs['euclidean_loss']
        epoch = all_data['epoch']
        all_data['loss_outputs'] = {
            'loss': (self.warmup[epoch - 1] * kld) + nll + (cross_entropy * self.CE_weight)
                                + self._lambda * euclidean_loss,
            'kld_loss': kld.item(),
            'nll_loss': nll.item(),
            'cross_entropy_loss': cross_entropy.item(),
            'euclidean_loss': euclidean_loss.item()
        }
        return all_data


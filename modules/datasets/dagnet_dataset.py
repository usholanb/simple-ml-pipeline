from typing import Dict
import csv

import pandas as pd

from modules.datasets.base_datasets.default_dataset import DefaultDataset
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from modules.helpers.csv_saver import CSVSaver
from utils.constants import DATA_DIR, PROJECT_DIR, PROCESSED_DATA_DIR
from utils.registry import registry


@registry.register_dataset('dagnet_dataset')
class DagnetDataset(DefaultDataset):

    def __init__(self, configs):
        super().__init__(configs)
        self.n_agents = self.configs.get('dataset').get('n_agents')

    def collect(self) -> None:
        path = f'{PROCESSED_DATA_DIR}/{self.name}.csv.gz'
        CSVSaver().clean_csv(path)

        for split_name, split_dir in self.configs.get('dataset').get('input_path').items():
            data_loader = self.read_source(f'{DATA_DIR}/{split_dir}')

            for index, (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
                obs_goals, pred_goals, seq_start_end) in enumerate(data_loader):
                obs = np.vstack([obs_traj, pred_traj])
                rel = np.vstack([obs_traj_rel, pred_traj_rel])
                goals = np.vstack([obs_goals, pred_goals])
                goals = goals.reshape((*goals.shape, 1))
                frames_n, batch10, x_y = obs.shape
                obs = obs.reshape((frames_n, self.n_agents, batch10 // self.n_agents, x_y))
                rel = rel.reshape((frames_n, self.n_agents, batch10 // self.n_agents, x_y))
                goals = goals.reshape((frames_n, self.n_agents, batch10 // self.n_agents))

                obs = obs.reshape(frames_n, -1)
                rel = rel.reshape(frames_n, -1)
                goals = goals.reshape(frames_n, -1)

                all_data = np.concatenate([obs, rel, goals], axis=1)
                df = pd.DataFrame(all_data)
                df.insert(loc=self.split_i, column='split', value=split_name)
                CSVSaver().append_rows(df, path)

    def read_source(self, split_dir):
        return dagnet_data_loader(split_dir)

    def get_dataloaders(self, configs):

        class LocalDagnetDataset(Dataset):
            """Dataloder for the Basketball trajectories datasets"""

            def __init__(self, df, configs):
                super(LocalDagnetDataset, self).__init__()
                self.configs = configs
                self.num_samples = len(df)
                self.n_agents = 10 if (configs.get('trainer').get('players') == 'all') else 5
                self.df = df

            def __len__(self):
                return self.num_samples

            def __max_agents__(self):
                # in bball the number of agents per scene is always the same
                return self.n_agents

            def __getitem__(self, idx):
                print('hi')

        df = CSVSaver().load(configs)
        train = LocalDagnetDataset(df.loc[df['split'] == 'train'], configs)
        valid = LocalDagnetDataset(df.loc[df['split'] == 'valid'], configs)
        test = LocalDagnetDataset(df.loc[df['split'] == 'test'], configs)

        n_max_agents = max(train.__max_agents__(), valid.__max_agents__(), test.__max_agents__())

        train_loader = DataLoader(
            train,
            **configs.get('dataset').get('train_dataloader'),
            collate_fn=seq_collate)

        valid_loader = DataLoader(
            valid,
            **configs.get('dataset').get('valid_dataloader'),
            collate_fn=seq_collate)

        test_loader = DataLoader(
            test,
            **configs.get('dataset').get('test_dataloader'),
            collate_fn=seq_collate)

        return train_loader, valid_loader, test_loader, n_max_agents


def dagnet_data_loader(set_path, shuffle=True):
    arguments = {
        'n_agents': 10,
        'obs_len': 10,
        'pred_len': 40,
        'num_workers': 4,
        'batch_size': 100,
    }
    dataset = BasketDataset(
        set_path,
        n_agents=arguments['n_agents'],
        obs_len=arguments['obs_len'],
        pred_len=arguments['pred_len']
    )
    loader = DataLoader(
        dataset,
        batch_size=arguments['batch_size'],
        shuffle=shuffle,
        num_workers=arguments['num_workers'],
        collate_fn=seq_collate)
    return loader


def _read_files(_path):
    sequences = np.load(f'{_path}/trajectories.npy')
    goals = np.load(f'{_path}/goals.npy')
    return sequences, goals


def seq_collate(data):
    (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
     obs_goals, pred_goals) = zip(*data)

    batch_size = len(obs_traj)
    obs_seq_len, n_agents, features = obs_traj[0].shape
    pred_seq_len, _, _ = pred_traj[0].shape

    obs_traj = torch.cat(obs_traj, dim=1)
    pred_traj = torch.cat(pred_traj, dim=1)
    obs_traj_rel = torch.cat(obs_traj_rel, dim=1)
    pred_traj_rel = torch.cat(pred_traj_rel, dim=1)
    obs_goals = torch.cat(obs_goals, dim=1)
    pred_goals = torch.cat(pred_goals, dim=1)

    # fixed number of agent for every play -> we can manually build seq_start_end
    idxs = list(range(0, (batch_size * n_agents) + n_agents, n_agents))
    seq_start_end = [[start, end] for start, end in zip(idxs[:], idxs[1:])]

    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, obs_goals,
        pred_goals, torch.tensor(seq_start_end)
    ]
    return tuple(out)


class BasketDataset(Dataset):
    """Dataloder for the Basketball trajectories datasets"""

    def __init__(self, path, n_agents, obs_len=10, pred_len=40):
        super(BasketDataset, self).__init__()

        self.data_dir = path
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.n_agents = n_agents

        # 'trajectories' shape (seq_len, batch*n_agents, 2) -> 2 = coords (x,y) for the single player
        # 'goals' shape (seq_len, batch*n_agents)
        traj_abs, goals = _read_files(self.data_dir)

        assert traj_abs.shape[0] == self.seq_len and goals.shape[0] == self.seq_len
        #assert self.seq_len <= traj_abs.shape[0] and self.seq_len <= goals.shape[0]

        num_seqs = traj_abs.shape[1] // self.n_agents
        idxs = [idx for idx in range(0, (num_seqs * self.n_agents) + n_agents, n_agents)]
        seq_start_end = [[start, end] for start, end in zip(idxs[:], idxs[1:])]

        self.num_samples = len(seq_start_end)

        traj_rel = np.zeros(traj_abs.shape)
        traj_rel[1:, :, :] = traj_abs[1:, :, :] - traj_abs[:-1, :, :]

        self.obs_traj = torch.from_numpy(traj_abs).type(torch.float)[:obs_len, :, :]
        self.obs_traj_rel = torch.from_numpy(traj_rel).type(torch.float)[:obs_len, :, :]
        self.obs_goals = torch.from_numpy(goals).type(torch.float)[:obs_len, :]
        self.pred_traj = torch.from_numpy(traj_abs).type(torch.float)[obs_len:, :, :]
        self.pred_traj_rel = torch.from_numpy(traj_rel).type(torch.float)[obs_len:, :, :]
        self.pred_goals = torch.from_numpy(goals).type(torch.float)[obs_len:, :]
        self.seq_start_end = seq_start_end

    def __len__(self):
        return self.num_samples

    def __max_agents__(self):
        # in bball the number of agents per scene is always the same
        return self.n_agents

    def __getitem__(self, idx):
        start, end = self.seq_start_end[idx]
        out = [
            self.obs_traj[:, start:end, :], self.pred_traj[:, start:end, :],
            self.obs_traj_rel[:, start:end, :], self.pred_traj_rel[:, start:end, :],
            self.obs_goals[:, start:end], self.pred_goals[:, start:end]
        ]

        return out



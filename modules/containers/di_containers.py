import torch
from dependency_injector import containers, providers
from modules.helpers.csv_saver import CSVSaver
from utils.registry import Registry


class SaverContainer(containers.DeclarativeContainer):
    csv_saver = providers.Singleton(
        CSVSaver
    )


class TrainerContainer(containers.DeclarativeContainer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")







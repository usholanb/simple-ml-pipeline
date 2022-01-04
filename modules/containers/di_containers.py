import torch
from dependency_injector import containers


class TrainerContainer(containers.DeclarativeContainer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







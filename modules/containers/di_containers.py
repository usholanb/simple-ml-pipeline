import torch


class TrainerContainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







from enum import Enum
import torch
import torch.nn as nn
from modules.models.base_models.torch_model import BaseTorchModel
from utils.registry import registry
from torch.functional import F


class LocalModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.layer1 = nn.Linear(self.input_dim, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 3)

    def forward(self, x, middle_feat_cum):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return middle_feat_cum, self.layer3(x)


@registry.register_model('dynamic_net')
class DynamicNet(BaseTorchModel):
    def __init__(self, special_inputs):
        super().__init__()
        self.__dict__.update(special_inputs)
        self.models = [LocalModel(self.input_dim)]
        self.boost_rate = nn.Parameter(torch.tensor(self.lr, requires_grad=True, device=self.device))

    def add(self, model):
        self.models.append(model)

    def parameters(self):
        params = []
        for m in self.models:
            params.extend(m.parameters())

        params.append(self.boost_rate)
        return params

    def to_cuda(self):
        for m in self.models:
            m.cuda()

    def to_eval(self):
        for m in self.models:
            m.eval()

    def to_train(self):
        for m in self.models:
            m.train(True)

    def forward(self, x):
        if len(self.models) == 0:
            return None
        middle_feat_cum = None
        prediction = None
        # with torch.no_grad():
        for m in self.models:
            if middle_feat_cum is None:
                middle_feat_cum, prediction = m(x, middle_feat_cum)
            else:
                middle_feat_cum, pred = m(x, middle_feat_cum)
                prediction += pred
        return self.boost_rate * prediction

    # def forward_grad(self, x):
    #     if len(self.models) == 0:
    #         return None
    #     # at least one model
    #     middle_feat_cum = None
    #     prediction = None
    #     for m in self.models:
    #         if middle_feat_cum is None:
    #             middle_feat_cum, prediction = m(x, middle_feat_cum)
    #         else:
    #             middle_feat_cum, pred = m(x, middle_feat_cum)
    #             prediction += pred
    #     return middle_feat_cum, self.boost_rate * prediction

    # @classmethod
    # def from_file(cls, path, builder):
    #     d = torch.load(path)
    #     net = DynamicNet()
    #     net.boost_rate = d['boost_rate']
    #     for stage, m in enumerate(d['models']):
    #         submod = builder(stage)
    #         submod.load_state_dict(m)
    #         net.add(submod)
    #     return net




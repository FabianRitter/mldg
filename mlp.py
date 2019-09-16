import torch.nn as nn
import torch.nn.functional as F

from ops import linear

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(60, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 1)

    def forward(self, x, meta_loss=None, meta_step_size=None, stop_gradient=False):
        x = linear(inputs=x,
                   weight=self.fc1.weight,
                   bias=self.fc1.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)
        x = F.relu(x)
        x = F.dropout(x, self.config['dropout'])

        x = linear(inputs=x,
                   weight=self.fc2.weight,
                   bias=self.fc2.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)
        x = F.relu(x)
        x = F.dropout(x, self.config['dropout'])

        x = linear(inputs=x,
                   weight=self.fc3.weight,
                   bias=self.fc3.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)

        return x.view(-1)
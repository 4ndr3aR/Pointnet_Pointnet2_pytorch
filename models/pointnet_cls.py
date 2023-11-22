import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class SigmoidRange(nn.Module):
    '''
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.


    Examples::

        >>> m = nn.Sigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''

    def __init__(self, low, high):
        super(SigmoidRange, self).__init__()
        self.low  = low
        self.high = high

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(input) * (self.high - self.low) + self.low 


class get_model(nn.Module):
    def __init__(self, out_features=40, normal_channel=True, y_range=None):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_features)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.y_range = y_range

        if self.y_range is not None:
            self.sigmoid_range = SigmoidRange(self.y_range[0], self.y_range[1])
        else:
            self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        if self.y_range is not None:
            x = self.sigmoid_range(x)
        else:
            x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, y_range=None, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.y_range = y_range

    def forward(self, pred, target, trans_feat):
        #print(f'self.y_range: {self.y_range}')
        if self.y_range is not None:
            pred = pred.squeeze(1)
            #print(f'pred  : {pred.shape} - pred: {pred}')
            #print(f'target: {target.shape} - target: {target}')
            loss = F.mse_loss(pred, target)
            #return loss
        else:
            loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

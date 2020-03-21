import torch
import torch.nn as nn
import torch.nn.functional as F

class SimLoss(nn.Module):

    def __init__(self, weight):
        super(SimLoss, self).__init__()
        self.weight = weight

    def forward(self, output1, output2, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)
        losses = self.weight * distances
        return losses.mean() if size_average else losses.sum()


#debugging
# output1 = torch.randn((10,9,256), dtype=torch.float)
# output2 = torch.randn((10,9,256), dtype=torch.float)
# simLoss = SimLoss(0.3)
#
# loss = simLoss(output1, output2)
# print(loss.item())
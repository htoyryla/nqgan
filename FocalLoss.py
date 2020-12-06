import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2, logits=False):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha #torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')            
        targets = targets.type(torch.long)
        #at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


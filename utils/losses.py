import torch
from torch.nn import functional as F
import torch.nn as nn
# class ContrastiveLoss(nn.Module):
#     """
#     Contrastive loss
#     Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
#     """

#     def __init__(self, margin):
#         super(ContrastiveLoss, self).__init__()
#         self.margin =margin
#         self.eps = 1e-9

#     def forward(self, output1, output2, target, size_average=True):
#         distances = (output2 - output1).pow(2).mean(1)  # squared distances
#         # print(distances.mean())
#         losses = 0.5 * (target.float() * distances +
#                         (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)/torch.clamp(distances.sqrt(),0, 1))  #/torch.clamp(distances.sqrt(),0, 1)
#         return losses.mean() if size_average else losses.sum()
        
class CosContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(CosContrastiveLoss, self).__init__()
        self.eps = 1e-9
        self.margin = margin
    def forward(self, output1, output2, target, size_average=True):

        cossim = torch.cosine_similarity(output1, output2, dim=1)
        distances = 1 - cossim

        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps)))  #/torch.clamp(distances.sqrt(),0, 1)

        return losses.mean()


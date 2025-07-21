import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Normalize các vector embedding
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Tính cosine similarity (dot product do đã normalize)
        pos_sim = (anchor * positive).sum(dim=1)
        neg_sim = (anchor * negative).sum(dim=1)

        # Triplet loss theo cosine similarity: muốn pos_sim cao, neg_sim thấp
        losses = F.relu(neg_sim - pos_sim + self.margin)

        return losses.mean()

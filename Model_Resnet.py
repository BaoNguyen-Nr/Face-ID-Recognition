import torch.nn as nn
import torchvision.models as models

class FaceNetEmbedding(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceNetEmbedding, self).__init__()
        base_model = models.resnet18(weights='IMAGENET1K_V1')
        modules = list(base_model.children())[:-1]  # Bỏ FC cuối
        self.feature_extractor = nn.Sequential(*modules)
        self.embedding = nn.Linear(512, embedding_size)  # Từ 512 -> 128

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x

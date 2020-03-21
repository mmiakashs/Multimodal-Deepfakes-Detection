from torchvision import models
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, original_model, embed_size):
        super(Encoder, self).__init__()
        self.embed_size = embed_size

        self.feature_extractor = nn.Sequential(*list(original_model.children())[:-2])
        num_ftrs = self.feature_extractor[-1].in_features
        self.fc = nn.Linear(num_ftrs, self.feature_embed_size)

        self.reset_parameters()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.)
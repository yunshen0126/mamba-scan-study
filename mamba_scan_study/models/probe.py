import torch.nn as nn


class LinearProbe(nn.Module):
    """Token-wise linear probe for frozen backbone features."""

    def __init__(self, d_model, n_classes):
        super().__init__()
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, token_features):
        return self.head(token_features)


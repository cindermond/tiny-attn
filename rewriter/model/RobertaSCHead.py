import torch
from torch import nn

class RobertaSCHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, output_nlayers, num_labels):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(output_nlayers)])
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = features[:, 0, :].view(features.size(dim=0),1,-1).squeeze(dim=1)
        for l in self.linears:
            x = self.dropout(x)
            x = l(x)
            x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
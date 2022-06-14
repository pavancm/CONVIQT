import torch.nn as nn

class GRUModel(nn.Module):
    """
    c_out inception_time output
    n_out model output
    """
    def __init__(self, c_in, hidden_size, projection_dim, normalize,\
                 num_layers = 1, **kwargs):
        super().__init__(**kwargs)
        self.gru = nn.GRU(c_in, hidden_size, batch_first = True, \
                          num_layers = num_layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
        )
        self.normalize = normalize
        self.c_out = hidden_size

    def forward(self, x_i, x_j):
        h_i, h_in = self.gru(x_i)
        h_j, h_jn = self.gru(x_j)
        
        h_i = self.gap(h_i.transpose(1,2))
        h_j = self.gap(h_j.transpose(1,2))
        
        h_i = h_i.view(-1, self.c_out)
        h_j = h_j.view(-1, self.c_out)
        
        if self.normalize:
            h_i = nn.functional.normalize(h_i, dim=1)
            h_j = nn.functional.normalize(h_j, dim=1)
        
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return z_i, z_j, h_i, h_j
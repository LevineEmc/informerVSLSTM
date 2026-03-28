import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, seq_len, pred_len, output_dim):
        super(LSTMModel, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim * seq_len, pred_len * output_dim)

    def forward(self, x):
        # x: [B, seq_len, input_dim]

        out, _ = self.lstm(x)

        # flatten
        out = out.reshape(out.shape[0], -1)

        out = self.fc(out)

        out = out.reshape(out.shape[0], self.pred_len, -1)

        return out
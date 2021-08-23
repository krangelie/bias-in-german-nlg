import torch
import torch.nn as nn
from torch import tanh
import pytorch_lightning as pl
from torch.nn import functional as F
from pytorch_lightning.metrics.functional import accuracy, f1

from src.classifier.torch_helpers.regard_classifier import RegardClassifier


class RegardLSTM(RegardClassifier):
    def __init__(
        self,
        n_embed,
        n_hidden,
        n_hidden_lin,
        n_output,
        n_layers,
        lr,
        weight_vector,
        bidirectional,
        gru,
        drop_p,
        drop_p_gru,
    ):
        RegardClassifier.__init__(
            self, n_embed, n_hidden_lin, n_output, lr, weight_vector, drop_p
        )
        drop_p_gru = drop_p_gru if drop_p_gru is not None else 0
        drop_p = drop_p if drop_p is not None else 0
        if gru:
            if n_hidden_lin > 0:

                self.lin1 = nn.Linear(n_embed, n_hidden_lin)
                self.dropout = nn.Dropout(drop_p)
                self.lstm = nn.GRU(
                    n_hidden_lin,
                    n_hidden,
                    n_layers,
                    batch_first=True,
                    dropout=drop_p_gru,
                    bidirectional=bidirectional,
                )

            else:
                self.lstm = nn.GRU(
                    n_embed,
                    n_hidden,
                    n_layers,
                    batch_first=True,
                    dropout=drop_p_gru,
                    bidirectional=bidirectional,
                )
        else:
            if n_hidden_lin > 0:
                self.lin1 = nn.Linear(n_embed, n_hidden_lin)
                self.lstm = nn.LSTM(
                    n_hidden_lin,
                    n_hidden,
                    n_layers,
                    batch_first=True,
                    dropout=drop_p_gru,
                    bidirectional=bidirectional,
                )
            else:
                self.lstm = nn.LSTM(
                    n_embed,
                    n_hidden,
                    n_layers,
                    batch_first=True,
                    dropout=drop_p_gru,
                    bidirectional=bidirectional,
                )
        self.fc = (
            nn.Linear(n_hidden * 2, n_output)
            if bidirectional
            else nn.Linear(n_hidden, n_output)
        )

    def forward(self, input_words):
        # INPUT   :  (batch_size, seq_length)
        if self.n_hidden_lin > 0:
            lin_out = self.lin1(input_words)
            lin_out = tanh(lin_out)
            lin_out = self.dropout(lin_out)
            lstm_out, h = self.lstm(lin_out)  # (batch_size, seq_length, n_hidden)
        else:
            lstm_out, h = self.lstm(input_words)
        fc_out = self.fc(lstm_out)  # (batch_size, seq_length, n_output)
        fc_out = fc_out[
            :, -1, :
        ]  # take only result of end of a sequence (batch_size, n_output)

        return fc_out, h

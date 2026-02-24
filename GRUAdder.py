import torch
import torch.nn as nn
import torch.nn.functional as F


class Adder(nn.Module):
    """
    Exact base-10 addition using a strict nn.RNNCell.
    Carry is stored as the hidden state.
    """

    def __init__(self):
        super().__init__()

        self.rnn = nn.RNNCell(
            input_size=2,
            hidden_size=1,
            nonlinearity='relu'
        )

        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():

            # Zero everything first
            for p in self.rnn.parameters():
                p.zero_()

            # Compute s = a + b + h

            # W_ih shape: (1, 2)
            self.rnn.weight_ih[0, 0] = 1.0  # a
            self.rnn.weight_ih[0, 1] = 1.0  # b

            # W_hh shape: (1, 1)
            self.rnn.weight_hh[0, 0] = 1.0  # carry

            # No bias needed

    def forward(self, a, b):
        """
        a, b: (batch, seq_len) digits 0-9
        """

        batch, seq_len = a.shape
        device = a.device

        h = torch.zeros(batch, 1, device=device)
        digits = []

        for i in reversed(range(seq_len)):

            x = torch.stack([a[:, i], b[:, i]], dim=1)

            # compute u = a + b + h
            u = self.rnn(x, h)

            # carry using ReLU threshold trick
            r1 = F.relu(u - 9)
            r2 = F.relu(u - 10)
            carry = r1 - r2

            s = a[:, i] + b[:, i] + h.squeeze(1)
            digit = s - 10 * carry.squeeze(1)

            digits.append(digit.unsqueeze(1))

            h = carry  # pass carry forward

        digits = torch.cat(digits[::-1], dim=1)

        return digits

"""
17-parameter RNN adder. Compiled from the addition algorithm — no training.

Hidden state h = [n1, n2] encodes carry via: carry = n1 - n2
    n1 = ReLU(s - 9)   }  two-neuron ReLU trick to replace
    n2 = ReLU(s - 10)  }  Heaviside: carry = n1 - n2 ∈ {0,1} exactly

Parameter breakdown:
    rnn.weight_ih_l0  (2,2)  [[1,1],[1,1]]    4
    rnn.weight_hh_l0  (2,2)  [[1,-1],[1,-1]]  4
    rnn.bias_ih_l0    (2,)   [0, 0]           2
    rnn.bias_hh_l0    (2,)   [-9, -10]        2
    carry_out.weight  (1,2)  [[1, -1]]        2
                                             ---
    nn.RNN mandatory bias_ih                  +2  (zeros, structural)
    output bias                               +1
                                             ---
    TOTAL                                    17
"""
import torch, torch.nn as nn


class RNNAdder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=2, hidden_size=2, nonlinearity='relu', batch_first=True)
        self.carry_out = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.rnn.weight_ih_l0.fill_(1.)
            self.rnn.weight_hh_l0.copy_(torch.tensor([[1.,-1.],[1.,-1.]]))
            self.rnn.bias_ih_l0.zero_()
            self.rnn.bias_hh_l0.copy_(torch.tensor([-9., -10.]))
            self.carry_out.weight.copy_(torch.tensor([[1., -1.]]))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a, b: (batch, N) float tensors of digits, most-significant first.
        Returns (batch, N+1) long tensor of result digits.
        """
        batch = a.shape[0]
        x = torch.stack([a, b], dim=2).flip(1)             # (batch, N, 2), LSB first
        h_seq, h_fin = self.rnn(x, torch.zeros(1, batch, 2))
        carry = self.carry_out(h_seq).squeeze(-1)           # (batch, N)
        carry_prev = torch.cat([torch.zeros(batch, 1), carry[:, :-1]], dim=1)
        s = a.flip(1) + b.flip(1) + carry_prev
        digits = (s - 10. * carry).round().long().flip(1)
        final_carry = self.carry_out(h_fin.squeeze(0)).round().long()
        return torch.cat([final_carry, digits], dim=1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

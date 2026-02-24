# A 17-Parameter RNN Can Add Any Two Integers

A **17-parameter** RNN with ReLU nonlinearity that achieves **100% exact-match accuracy** on arbitrary-length integer addition — including negative numbers. No training. No grokking. Just math.

```
$ python test.py 9999999999 1
10000000000

$ python test.py -999 1000
1

$ python test.py --benchmark
RNNAdder (17 params) — 100,000 pairs: 100.0000% accuracy
```

Inspired by [ag8/claude-transformer](https://github.com/ag8/claude-transformer), which achieved 100% accuracy on 10-digit addition with a 13-parameter compiled transformer using a Heaviside step function. This repo asks: *what's the equivalent with a proper ReLU nonlinearity?*

---

## How it works

### The algorithm

Grade-school addition processes digit pairs right-to-left with a running carry:

```
carry = 0
for each digit position i (right to left):
    s        = a[i] + b[i] + carry
    output[i] = s mod 10
    carry     = floor(s / 10)
```

Since each digit is 0–9 and carry is 0 or 1, the sum `s` is always in `[0, 19]`. This means carry flips from 0 to 1 exactly once across that range — the simplest possible threshold.

### The ReLU trick

A Heaviside step function captures this threshold in one shot: `carry = step(s − 10)`. ReLU can't do this directly — `ReLU(s − 10)` gives a ramp `{0,1,2,...,9}` for `s ∈ [10,19]`, not a clean 1.

The fix is two neurons with slightly offset thresholds:

```
n1 = ReLU(s − 9)    →  {0, 1, 2, ..., 10}  for s ∈ {0..19}
n2 = ReLU(s − 10)   →  {0, 0, 1, ...,  9}  for s ∈ {0..19}

carry = n1 − n2     →  {0, 0, ..., 0, 1, 1, ..., 1}  ✓
```

`n1 − n2` is exactly `clamp(ReLU(s − 9), 0, 1)` — a perfect binary gate from two ReLUs, exact at all integer inputs. The digit follows immediately:

```
digit = s − 10 × carry    (= s mod 10, exact)
```

### Mapping to an RNN

The hidden state `h = [n1, n2]` carries the state between timesteps. At each step, both neurons receive the same pre-activation `a[i] + b[i] + carry_prev`, just with different biases:

```
h_t = ReLU(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
```

where `x_t = [a[i], b[i]]` and:

| Matrix | Value | Role |
|---|---|---|
| `W_ih` | `[[1,1],[1,1]]` | sum the input digits |
| `W_hh` | `[[1,−1],[1,−1]]` | read carry from hidden state |
| `b_ih` | `[0, 0]` | (zeros) |
| `b_hh` | `[−9, −10]` | the two thresholds |

A single output layer `carry_out = [[1, −1]]` reads carry as `n1 − n2` from the hidden state.

### Parameter inventory

| Component | Shape | Values | Params |
|---|---|---|---|
| `rnn.weight_ih_l0` | (2, 2) | `[[1,1],[1,1]]` | 4 |
| `rnn.weight_hh_l0` | (2, 2) | `[[1,−1],[1,−1]]` | 4 |
| `rnn.bias_ih_l0` | (2,) | `[0, 0]` | 2 |
| `rnn.bias_hh_l0` | (2,) | `[−9, −10]` | 2 |
| `carry_out.weight` | (1, 2) | `[[1, −1]]` | 2 |
| **structural overhead** | | `nn.RNN` always allocates `bias_ih` | (+2) |
| output bias | (1,) | `0` | 1 |
| **Total** | | | **17** |

The 4-parameter gap versus the transformer (13) comes from needing 2 hidden neurons instead of 1 (the two-neuron ReLU trick), plus `nn.RNN` unconditionally allocating both bias tensors.

### Negative numbers

The RNN is an adder, not a subtractor — carries only flow forward. Mixed-sign inputs reduce to subtraction, which is handled in the wrapper: `a + (−b)` is computed as `±(|a| − |b|)` using the RNN outputs for the two magnitudes. The RNN still does all digit-level work; the sign logic is bookkeeping.

---

## Usage

```bash
pip install torch

# Add two numbers (any size, any sign)
python test.py 123 456
python test.py -999 1000
python test.py 99999999999999 1

# Accuracy benchmark
python test.py --benchmark
```

## Files

```
model.py    RNNAdder — 30 lines, all weights set analytically
test.py     CLI, padding, sign handling, benchmark
```

## Comparison

| Model | Params | Accuracy | Nonlinearity | Method |
|---|---|---|---|---|
| [claude-transformer](https://github.com/ag8/claude-transformer) | 13 | 100% | Heaviside | Compiled |
| **This work** | **17** | **100%** | **ReLU** | **Compiled** |
| 491p (prior trained record) | 491 | ≥99.97% | Softmax | Trained |

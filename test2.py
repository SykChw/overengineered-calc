#!/usr/bin/env python3
"""
Usage:
    python test.py 123 456              -> 579
    python test.py -999 1000            -> 1
    python test.py 9999999999 1         -> 10000000000
    python test.py --benchmark          -> accuracy over 100k random pairs
"""
import sys, random, argparse
import torch
from model import RNNAdder

model = RNNAdder()
model.eval()


def digits(n: int, width: int) -> torch.Tensor:
    return torch.tensor([[int(d) for d in str(n).zfill(width)]], dtype=torch.float32)


def rnn_add_positive(a: int, b: int) -> int:
    w = max(len(str(a)), len(str(b)))
    out = model(digits(a, w), digits(b, w))
    return int(''.join(map(str, out[0].tolist())))


def add(a: int, b: int) -> int:
    if a >= 0 and b >= 0: return rnn_add_positive(a, b)
    if a < 0  and b < 0:  return -rnn_add_positive(-a, -b)
    # mixed signs: a + b = sign(|a|-|b|) * rnn_add_positive(big-small)
    pos, neg = (a, -b) if a >= 0 else (b, -a)
    return rnn_add_positive(pos, 0) - rnn_add_positive(neg, 0) if pos >= neg \
        else -(rnn_add_positive(neg, 0) - rnn_add_positive(pos, 0))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('numbers', nargs='*', type=int)
    p.add_argument('--benchmark', action='store_true')
    args = p.parse_args()

    if args.benchmark:
        n = 100_000
        ok = sum(
            add(a := random.randint(-9_999_999_999, 9_999_999_999),
                b := random.randint(-9_999_999_999, 9_999_999_999)) == a + b
            for _ in range(n)
        )
        print(f"RNNAdder ({model.n_params()} params) — {n:,} pairs: {100*ok/n:.4f}% accuracy")
    elif len(args.numbers) == 2:
        a, b = args.numbers
        print(add(a, b))
    else:
        p.print_help()

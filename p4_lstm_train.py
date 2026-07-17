#!/usr/bin/env python3
"""
p4_lstm_train.py -- train a next-item LSTM on a trace PREFIX and dump per-position
top-M predictions for the WHOLE trace, for replay by p4_strongbar.py's LSTM arm.

Leakage discipline (matches the Markov convention exactly):
  * vocabulary and model weights come from the first --train-frac of the trace ONLY;
  * inference then runs causally over all positions -- the prediction at position i
    conditions on requests <= i and never on future requests or on cache state.
Like the Markov tables, the model is also applied inside its own training prefix;
identical convention on both arms keeps the bar comparison fair.

Honest-scope notes (state these in the paper):
  * the model predicts the NEXT request; Markov tables score "appears within window=16".
    This slightly under-credits the LSTM on lagged patterns -- acceptable for a bar that
    is a max over arms, since Markov-1/2/3 remain in the sweep;
  * objects outside the top --vocab training objects map to UNK and are never prefetched
    (UNK targets are also excluded from the loss); coverage is printed -- check it.

Typical (DGX, one GPU):  ~5-10 min train + ~1-2 min dump per trace.
    python p4_lstm_train.py --trace data/wiki_2019t.oracleGeneral --limit 2000000 \
        --out preds/wiki_2019t.lstm.npz
"""
from __future__ import annotations

import argparse
import os
import time
from collections import Counter

import numpy as np

from p4_cache import load_oracle_general


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--vocab", type=int, default=100_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=96, help="TBPTT window")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--top-m", type=int, default=16, help="candidates dumped per position")
    ap.add_argument("--chunk", type=int, default=4096, help="inference chunk (positions)")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    trace = load_oracle_general(args.trace, limit=args.limit)
    ids = trace["obj_id"].astype(np.int64)
    n = int(trace["n"])
    cut = int(n * args.train_frac)

    # ---- vocab from the train prefix only; class 0 = UNK ----------------------------------
    freq = Counter(ids[:cut].tolist())
    vocab = [o for o, _ in freq.most_common(args.vocab)]
    cls_of = {o: c for c, o in enumerate(vocab, start=1)}
    vocab_arr = np.full(len(vocab) + 1, -1, dtype=np.int64)   # class -> obj id; UNK -> -1
    for o, c in cls_of.items():
        vocab_arr[c] = o
    V = len(vocab) + 1
    x_all = np.fromiter((cls_of.get(int(o), 0) for o in ids), dtype=np.int64, count=n)
    cov = float((x_all[:cut] != 0).mean())
    print(f"[vocab] {V-1:,} objects, train-token coverage {cov:.3f} "
          f"({'OK' if cov >= 0.5 else 'LOW -- consider --vocab higher'})")

    # ---- model ----------------------------------------------------------------------------
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, args.embed)
            self.lstm = nn.LSTM(args.embed, args.hidden, num_layers=args.layers,
                                batch_first=True)
            self.out = nn.Linear(args.hidden, V)

        def forward(self, x, h=None):
            y, h = self.lstm(self.emb(x), h)
            return y, h

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss(ignore_index=0)               # UNK never a target
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[model] V={V:,} embed={args.embed} hidden={args.hidden} params={n_par/1e6:.1f}M "
          f"on {dev}")

    # ---- training: contiguous rows + truncated BPTT ----------------------------------------
    B, S = args.batch, args.seq
    L = (cut - 1) // B
    if L < S:
        raise SystemExit("train prefix too short for batch/seq; lower --batch or --seq")
    xr = torch.from_numpy(x_all[:B * L].reshape(B, L))
    yr = torch.from_numpy(x_all[1:B * L + 1].reshape(B, L))
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda"))
    t0, step = time.time(), 0
    for ep in range(args.epochs):
        h, ep_loss, ep_batches = None, 0.0, 0
        for s in range(0, L - 1, S):
            xb = xr[:, s:s + S].to(dev, non_blocking=True)
            yb = yr[:, s:s + S].to(dev, non_blocking=True)
            with amp:
                y, h = net(xb, h)
                loss = lossf(net.out(y).reshape(-1, V), yb.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            h = tuple(t.detach() for t in h)
            ep_loss += loss.item(); ep_batches += 1; step += 1
        print(f"[train] epoch {ep+1}/{args.epochs}  loss {ep_loss/max(ep_batches,1):.3f}  "
              f"({time.time()-t0:.0f}s)")

    # ---- inference: one causal pass over the whole trace, dump top-M per position ----------
    # Row i = distribution over request i+1 given requests <= i. UNK prob is zeroed after
    # softmax (never suggested); remaining probs are conservative (slightly under-normalized).
    net.eval()
    M = args.top_m
    cand = np.full((n, M), -1, dtype=np.int64)
    prob = np.zeros((n, M), dtype=np.float16)
    t1, h = time.time(), None
    with torch.no_grad():
        for s in range(0, n, args.chunk):
            xb = torch.from_numpy(x_all[s:s + args.chunk]).view(1, -1).to(dev)
            with amp:
                y, h = net(xb, h)
                logits = net.out(y[0])
            p = logits.float().softmax(-1)
            p[:, 0] = 0.0
            pv, pi = p.topk(M, dim=-1)
            cand[s:s + xb.shape[1]] = vocab_arr[pi.cpu().numpy()]
            prob[s:s + xb.shape[1]] = pv.cpu().numpy().astype(np.float16)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, cand=cand, prob=prob)
    # quick self-diagnostic: top-1 next-request accuracy on the EVAL half (prediction at i
    # vs actual request i+1) -- a cheap signal of whether the model learned anything.
    nxt = x_all[cut + 1:n]
    pred_obj = cand[cut:n - 1, 0]
    actual_obj = np.where(nxt != 0, vocab_arr[nxt], -2)       # -2: never matches -1
    acc = float((pred_obj == actual_obj).mean())
    print(f"[dump] {args.out}  rows={n:,} M={M}  ({time.time()-t1:.0f}s)")
    print(f"[diag] eval-half top-1 next-request accuracy: {acc:.4f} "
          f"(vs ~0 for a broken model; compare with Markov pf precision, not directly OHR)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnostics for UTTT checkpoint collapse/regression investigations."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import alphazero
import neural_net
from game_runner import glob_file_triples, load_compressed


def _iter_from_path(path: str | Path) -> int | None:
    name = Path(path).name
    m = re.match(r"^(\d{4})-", name)
    return int(m.group(1)) if m else None


def _checkpoint_path(exp_dir: Path, iteration: int) -> Path:
    matches = sorted((exp_dir / "checkpoint").glob(f"{iteration:04d}-*.pt"))
    if not matches:
        raise FileNotFoundError(f"missing checkpoint {iteration:04d} under {exp_dir / 'checkpoint'}")
    return matches[-1]


def _load_checkpoint(path: Path) -> neural_net.NNWrapper:
    nn = neural_net.NNWrapper.load_checkpoint(
        alphazero.UltimateTicTacToeGS, str(path.parent), path.name
    )
    nn.nnet.eval()
    return nn


def _load_fixed_sample(exp_dir: Path, max_iter: int, sample_size: int, seed: int):
    triples = []
    for triple in glob_file_triples(str(exp_dir / "history"), "*-canonical-*.ptz"):
        it = _iter_from_path(triple[0])
        if it is not None and it <= max_iter:
            triples.append(triple)
    if not triples:
        raise RuntimeError(f"no history triples found for iter <= {max_iter} in {exp_dir / 'history'}")
    triples.sort(key=lambda t: (int(_iter_from_path(t[0]) or -1), t[0]))

    c_parts, v_parts, pi_parts = [], [], []
    remaining = sample_size
    rng = np.random.default_rng(seed)
    for c_path, v_path, pi_path, size in triples:
        if remaining <= 0:
            break
        c = load_compressed(c_path)
        v = load_compressed(v_path)
        pi = load_compressed(pi_path)
        n = min(int(c.shape[0]), int(v.shape[0]), int(pi.shape[0]))
        take = min(n, remaining)
        if take < n:
            idx = np.sort(rng.choice(n, size=take, replace=False))
            c = c[idx]
            v = v[idx]
            pi = pi[idx]
        else:
            c = c[:take]
            v = v[:take]
            pi = pi[:take]
        c_parts.append(c)
        v_parts.append(v)
        pi_parts.append(pi)
        remaining -= take

    canonical = torch.cat(c_parts, dim=0).float()
    target_v = torch.cat(v_parts, dim=0).float()
    target_pi = torch.cat(pi_parts, dim=0).float()
    return canonical, target_v, target_pi


def _predict_policy(nn: neural_net.NNWrapper, canonical: torch.Tensor, batch_size: int):
    ds = TensorDataset(canonical)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    policies = []
    values = []
    with torch.no_grad():
        for (cb,) in loader:
            cb = cb.float().contiguous().to(nn.device, non_blocking=nn._non_blocking)
            cb = nn._adapt_input_planes(cb)
            out_v, out_pi = nn.nnet(cb)
            policies.append(torch.exp(out_pi).detach().cpu())
            values.append(torch.exp(out_v).detach().cpu())
    return torch.cat(values, dim=0), torch.cat(policies, dim=0)


def _loss_and_policy(args):
    exp_dir = Path(args.exp_dir)
    canonical, target_v, target_pi = _load_fixed_sample(
        exp_dir, args.sample_max_iter, args.sample_size, args.seed
    )
    ds = TensorDataset(canonical, target_v, target_pi)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    rows = []
    base_policy = None
    base_value = None
    eps = 1e-12
    target_entropy = -(target_pi * torch.log(target_pi.clamp_min(eps))).sum(dim=1)
    target_top1 = target_pi.argmax(dim=1)

    for iteration in args.iters:
        ckpt = _checkpoint_path(exp_dir, iteration)
        nn = _load_checkpoint(ckpt)
        v_loss, pi_loss = nn.losses(loader)
        pred_v, pred_pi = _predict_policy(nn, canonical, args.batch_size)

        entropy = -(pred_pi * torch.log(pred_pi.clamp_min(eps))).sum(dim=1)
        confidence = pred_pi.max(dim=1).values
        top1 = pred_pi.argmax(dim=1)
        row = {
            "iter": iteration,
            "checkpoint": str(ckpt),
            "value_loss": float(v_loss),
            "policy_loss": float(pi_loss),
            "total_loss": float(v_loss + pi_loss),
            "policy_entropy_mean": float(entropy.mean()),
            "policy_confidence_mean": float(confidence.mean()),
            "target_policy_entropy_mean": float(target_entropy.mean()),
            "target_top1_agreement": float((top1 == target_top1).float().mean()),
            "value_win_prob_mean": float(pred_v[:, 0].mean()),
            "value_loss_prob_mean": float(pred_v[:, 1].mean()),
            "value_draw_prob_mean": float(pred_v[:, 2].mean()),
        }
        if base_policy is None:
            base_policy = pred_pi
            base_value = pred_v
            row["kl_from_base"] = 0.0
            row["top1_agreement_with_base"] = 1.0
            row["value_l1_from_base"] = 0.0
        else:
            kl = (base_policy * (torch.log(base_policy.clamp_min(eps)) - torch.log(pred_pi.clamp_min(eps)))).sum(dim=1)
            row["kl_from_base"] = float(kl.mean())
            row["top1_agreement_with_base"] = float((top1 == base_policy.argmax(dim=1)).float().mean())
            row["value_l1_from_base"] = float((pred_v - base_value).abs().sum(dim=1).mean())
        rows.append(row)

    return {
        "sample_size": int(canonical.shape[0]),
        "sample_max_iter": args.sample_max_iter,
        "seed": args.seed,
        "rows": rows,
    }


def _gate_summary(args):
    exp_dir = Path(args.exp_dir)
    wins = np.genfromtxt(exp_dir / "wins.csv", delimiter=",")
    draws = np.genfromtxt(exp_dir / "draws.csv", delimiter=",")
    wr = np.genfromtxt(exp_dir / "win_rate.csv", delimiter=",")
    rows = []
    for i in args.iters:
        for j in args.iters:
            if i == j or i >= wins.shape[0] or j >= wins.shape[1]:
                continue
            wi = int(wins[i, j])
            wj = int(wins[j, i])
            d = int(draws[i, j])
            games = wi + wj + d
            if games <= 0:
                continue
            rows.append({
                "i": i,
                "j": j,
                "games": games,
                "i_wins": wi,
                "j_wins": wj,
                "draws": d,
                "i_score": float((wi + 0.5 * d) / games),
                "win_rate_csv": None if np.isnan(wr[i, j]) else float(wr[i, j]),
            })
    return {"rows": rows}


def _write_outputs(result: dict, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    md = out.with_suffix(".md")
    lines = []
    if "sample_size" in result:
        lines.append(f"# Fixed-Sample Loss And Policy Drift\n")
        lines.append(f"sample_size={result['sample_size']} sample_max_iter={result['sample_max_iter']} seed={result['seed']}\n")
        lines.append("| iter | v_loss | pi_loss | total | H(policy) | conf | KL from base | top1 vs base | top1 vs target | value W/L/D |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in result["rows"]:
            lines.append(
                f"| {r['iter']} | {r['value_loss']:.4f} | {r['policy_loss']:.4f} | {r['total_loss']:.4f} | "
                f"{r['policy_entropy_mean']:.3f} | {r['policy_confidence_mean']:.3f} | {r['kl_from_base']:.4f} | "
                f"{r['top1_agreement_with_base']:.3f} | {r['target_top1_agreement']:.3f} | "
                f"{r['value_win_prob_mean']:.3f}/{r['value_loss_prob_mean']:.3f}/{r['value_draw_prob_mean']:.3f} |"
            )
    else:
        lines.append("# Gate Evidence Summary\n")
        lines.append("| i | j | games | i_wins | j_wins | draws | i_score | csv |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in result["rows"]:
            csv = "" if r["win_rate_csv"] is None else f"{r['win_rate_csv']:.3f}"
            lines.append(
                f"| {r['i']} | {r['j']} | {r['games']} | {r['i_wins']} | {r['j_wins']} | "
                f"{r['draws']} | {r['i_score']:.3f} | {csv} |"
            )
    md.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("loss-policy")
    lp.add_argument("--exp-dir", required=True)
    lp.add_argument("--iters", nargs="+", type=int, required=True)
    lp.add_argument("--sample-max-iter", type=int, default=77)
    lp.add_argument("--sample-size", type=int, default=4096)
    lp.add_argument("--batch-size", type=int, default=512)
    lp.add_argument("--seed", type=int, default=0)
    lp.add_argument("--out", required=True)

    gs = sub.add_parser("gate-summary")
    gs.add_argument("--exp-dir", required=True)
    gs.add_argument("--iters", nargs="+", type=int, required=True)
    gs.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.cmd == "loss-policy":
        result = _loss_and_policy(args)
    elif args.cmd == "gate-summary":
        result = _gate_summary(args)
    else:
        raise AssertionError(args.cmd)
    _write_outputs(result, Path(args.out))


if __name__ == "__main__":
    main()

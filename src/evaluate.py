"""원전 조건별 강건성 평가: 같은 검증셋에 조건을 입혀 IoU/F1 하락을 측정한다.

사용 예:
    python -m src.evaluate --ckpt runs/baseline/best.pt --split runs/baseline/split.json \
        --masks /content/data/masks --out runs/baseline/robustness.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

from src.augment.nuclear_conditions import CONDITIONS, apply_condition
from src.dataset import load_pair, to_tensor
from src.metrics import batch_stats, summarize


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(ck["encoder"], encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    size = ck["size"]

    val_imgs = [Path(s) for s in json.loads(a.split.read_text())["val"]]
    masks = {m.stem: m for m in a.masks.rglob("*.png")}
    pairs = [(i, masks[i.stem]) for i in val_imgs if i.stem in masks]

    rows = []
    for cond in CONDITIONS:
        for level in ([1] if cond == "clean" else a.levels):
            rng = np.random.default_rng(a.seed)  # 조건마다 같은 난수열 -> 비교 공정
            stats = None
            with torch.no_grad():
                for ip, mp in pairs:
                    img, mask = load_pair(ip, mp, size)
                    img, mask = apply_condition(cond, img, mask, rng, level=level)
                    x, y = to_tensor(img, mask)
                    pred = torch.sigmoid(model(x[None].to(dev))).cpu()[0] > 0.5
                    stats = batch_stats(pred, y.bool(), stats)
            m = summarize(stats)
            rows.append({"condition": cond, "level": level, **{k: round(v, 4) for k, v in m.items()}})
            print(f"{cond:13s} L{level}  IoU {m['iou']:.4f}  F1 {m['f1']:.4f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"저장 -> {a.out}")


if __name__ == "__main__":
    main()

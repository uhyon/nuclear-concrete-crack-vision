"""균열 분할 베이스라인 학습 (U-Net + ResNet34, Colab 무료 GPU 기준).

사용 예:
    python -m src.train --images /content/data/images --masks /content/data/masks \
        --out runs/baseline --epochs 15 --limit 10000

원전 조건 증강을 켠 회복 실험:
    python -m src.train ... --out runs/condition_aug --condition-aug
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, get_worker_info

from src.dataset import CrackDataset, find_pairs, split_pairs
from src.metrics import batch_stats, summarize


def seed_worker(_worker_id: int) -> None:
    """워커마다(그리고 epoch마다) 다른 난수열을 쓰게 해 증강이 겹치지 않도록 한다."""
    info = get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/baseline"))
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--limit", type=int, default=None, help="사용할 이미지 수 상한(무료 Colab용)")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--encoder-weights", default="imagenet", help='사전학습 가중치. "none"이면 처음부터 학습')
    p.add_argument("--condition-aug", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(a.images, a.masks)
    if not pairs:
        raise SystemExit("이미지-마스크 쌍을 찾지 못했습니다. 파일 이름이 같은지 확인하세요.")
    train_pairs, val_pairs = split_pairs(pairs, limit=a.limit)
    (a.out / "split.json").write_text(
        json.dumps({"train": [str(i) for i, _ in train_pairs], "val": [str(i) for i, _ in val_pairs]}, ensure_ascii=False)
    )
    print(f"train {len(train_pairs)} / val {len(val_pairs)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    weights = None if a.encoder_weights.lower() == "none" else a.encoder_weights
    model = smp.Unet(a.encoder, encoder_weights=weights, in_channels=3, classes=1).to(dev)
    dice = smp.losses.DiceLoss("binary")
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    tl = DataLoader(CrackDataset(train_pairs, a.size, train=True, condition_aug=a.condition_aug),
                    batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                    worker_init_fn=seed_worker)
    vl = DataLoader(CrackDataset(val_pairs, a.size), batch_size=a.batch, num_workers=a.workers)

    best, history = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                logits = model(x)
                loss = dice(logits, y) + bce(logits, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        sched.step()

        model.eval()
        stats = None
        with torch.no_grad():
            for x, y in vl:
                pred = torch.sigmoid(model(x.to(dev))) > 0.5
                stats = batch_stats(pred.cpu(), y.bool(), stats)
        m = summarize(stats)
        history.append({"epoch": ep, "loss": total / max(1, len(tl)), **m})
        print(f"[{ep}/{a.epochs}] loss {history[-1]['loss']:.4f}  IoU {m['iou']:.4f}  F1 {m['f1']:.4f}")
        if m["iou"] > best:
            best = m["iou"]
            torch.save({"model": model.state_dict(), "encoder": a.encoder, "size": a.size}, a.out / "best.pt")
    (a.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val IoU {best:.4f} -> {a.out / 'best.pt'}")


if __name__ == "__main__":
    main()

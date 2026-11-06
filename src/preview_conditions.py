"""한 장의 이미지에 원전 조건을 입힌 비교 그림을 만든다 (README용).

python -m src.preview_conditions --image sample.jpg --mask sample.png --out docs/conditions.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.augment.nuclear_conditions import CONDITIONS, apply_condition


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--mask", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("docs/conditions.png"))
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--level", type=int, default=2)
    a = p.parse_args()

    img = cv2.resize(cv2.imread(str(a.image)), (a.size, a.size))
    mask = (cv2.resize(cv2.imread(str(a.mask), 0), (a.size, a.size), interpolation=cv2.INTER_NEAREST)
            if a.mask else np.zeros((a.size, a.size), np.uint8))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(3 * len(CONDITIONS), 3.2))
    for ax, cond in zip(axes, CONDITIONS):
        out, m = apply_condition(cond, img.copy(), mask.copy(), rng, level=a.level)
        ax.imshow(out[..., ::-1])
        if a.mask:
            ax.contour(m > 0, levels=[0.5], colors="r", linewidths=0.6)
        ax.set_title(cond, fontsize=10)
        ax.axis("off")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print(f"저장 -> {a.out}")


if __name__ == "__main__":
    main()

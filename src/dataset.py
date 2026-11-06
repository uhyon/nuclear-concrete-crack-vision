"""이미지-마스크 쌍 데이터셋과 train/val 분할."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    """같은 파일 이름(확장자 제외)을 가진 이미지와 마스크를 짝짓는다."""
    masks = {p.stem: p for p in masks_dir.rglob("*.png")}
    pairs = []
    for ip in sorted(images_dir.rglob("*")):
        if ip.suffix.lower() in IMG_EXTS and ip.stem in masks:
            pairs.append((ip, masks[ip.stem]))
    return pairs


def split_pairs(pairs, val_ratio: float = 0.15, seed: int = 42, limit: int | None = None):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    if limit:
        pairs = pairs[:limit]
    n_val = max(1, int(len(pairs) * val_ratio))
    return pairs[n_val:], pairs[:n_val]


def load_pair(img_path: Path, mask_path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        raise FileNotFoundError(f"{img_path} / {mask_path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    # 얇은 균열이 리사이즈로 사라지지 않도록 최댓값 풀링 방식으로 줄인다.
    if mask.shape[0] > size * 2:
        k = mask.shape[0] // size
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, mask


try:
    import torch
    from torch.utils.data import Dataset

    from src.augment.nuclear_conditions import apply_condition

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def to_tensor(img: np.ndarray, mask: np.ndarray):
        x = img[..., ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        x = (x - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy((mask > 0).astype(np.float32))[None]

    class CrackDataset(Dataset):
        """condition_aug=True면 학습 중 원전 조건 열화를 무작위로 섞는다(회복 실험용)."""

        def __init__(self, pairs, size=512, train=False, condition_aug=False, aug_prob=0.5, seed=0):
            self.pairs, self.size, self.train = pairs, size, train
            self.condition_aug, self.aug_prob = condition_aug, aug_prob
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, i):
            img, mask = load_pair(*self.pairs[i], self.size)
            if self.train:
                if self.rng.random() < 0.5:
                    img, mask = img[:, ::-1], mask[:, ::-1]
                if self.rng.random() < 0.5:
                    img, mask = img[::-1], mask[::-1]
                img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
                if self.condition_aug and self.rng.random() < self.aug_prob:
                    name = self.rng.choice(["radiation", "low_light", "uneven_light", "motion_blur", "narrow_fov"])
                    img, mask = apply_condition(str(name), img, mask, self.rng, level=int(self.rng.integers(1, 4)))
            return to_tensor(img, mask)

except ImportError:  # torch 없이 변환 스크립트만 쓸 때
    pass

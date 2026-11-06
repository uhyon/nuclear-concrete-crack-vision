"""AI Hub 'SOC 시설물 균열패턴 이미지' 라벨(JSON)을 분할 마스크(PNG)로 변환한다.

라벨 구조(AI Hub 메타데이터 구조표 기준, 실제 파일로 확인 필요):
    image.name, image.width, image.height, image.object_included,
    image.annotations[] = {label, points, shape("Polygon"|"Polyline"), width?, px?}

- Polygon: 내부를 채운다.
- Polyline(주로 균열): 선으로 그린다. 두께는 라벨의 px(균열 폭 픽셀)가 있으면 그 값을,
  없으면 --line-width 기본값을 쓴다.

사용 예:
    python -m src.data.aihub_to_masks \
        --labels /content/data/labels --images /content/data/images \
        --out /content/data/masks --mode binary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# AI Hub 결함 10종. 인덱스 0은 배경.
CLASSES = [
    "background",
    "crack",
    "reticular crack",
    "detachment",
    "spalling",
    "efflorescence",
    "leak",
    "rebar",
    "material separation",
    "exhilaration",  # AI Hub 표기 그대로 (들뜸)
    "damage",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

# binary 모드에서 '균열'로 볼 클래스
CRACK_LABELS = {"crack", "reticular crack"}


def _get(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def parse_points(raw) -> np.ndarray:
    """[[x, y], ...] 또는 [x1, y1, x2, y2, ...] 또는 [[[x, y], ...]] 를 (N, 2) int 배열로."""
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 3:  # [[[x, y], ...]]
        arr = arr.reshape(-1, 2)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return np.round(arr).astype(np.int32)


def load_annotations(label: dict) -> tuple[dict, list[dict]]:
    image = _get(label, "image", "images", default={}) or {}
    if isinstance(image, list):
        image = image[0] if image else {}
    anns = _get(image, "annotations", default=None)
    if anns is None:
        anns = _get(label, "annotations", default=[]) or []
    return image, anns


def render_mask(
    label: dict,
    mode: str = "binary",
    line_width: int = 3,
    size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, str]:
    """라벨 dict 하나로 마스크를 만든다. (mask, image_name) 반환.

    mode="binary": 균열=1, 나머지=0
    mode="multiclass": CLASSES 인덱스
    """
    image, anns = load_annotations(label)
    name = _get(image, "name", "file_name", "filename", default="")
    if size is not None:
        h, w = size
    else:
        w = int(_get(image, "width", default=0))
        h = int(_get(image, "height", default=0))
    if not (h and w):
        raise ValueError(f"이미지 크기를 알 수 없음: {name}")

    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        label_name = str(_get(ann, "label", default="")).strip().lower()
        if mode == "binary":
            if label_name not in CRACK_LABELS:
                continue
            value = 1
        else:
            value = CLASS_TO_ID.get(label_name)
            if value is None:
                continue

        pts = parse_points(_get(ann, "points", default=[]))
        if len(pts) < 2:
            continue
        shape = str(_get(ann, "shape", default="Polyline")).lower()
        if shape == "polygon" and len(pts) >= 3:
            cv2.fillPoly(mask, [pts], int(value))
        else:
            px = _get(ann, "px", default=None)
            try:
                thick = max(1, int(round(float(px)))) if px not in (None, "") else line_width
            except (TypeError, ValueError):
                thick = line_width
            cv2.polylines(mask, [pts], isClosed=False, color=int(value), thickness=thick)
    return mask, name


def convert_dir(labels_dir: Path, images_dir: Path | None, out_dir: Path, mode: str, line_width: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for jp in sorted(labels_dir.rglob("*.json")):
        with open(jp, encoding="utf-8") as f:
            label = json.load(f)
        size = None
        image, _ = load_annotations(label)
        name = _get(image, "name", "file_name", "filename", default=jp.stem)
        if images_dir is not None and not (_get(image, "width") and _get(image, "height")):
            matches = list(images_dir.rglob(Path(name).name))
            if matches:
                img = cv2.imread(str(matches[0]), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    size = img.shape[:2]
        try:
            mask, name = render_mask(label, mode=mode, line_width=line_width, size=size)
        except ValueError as e:
            print(f"[skip] {jp.name}: {e}")
            continue
        stem = Path(name).stem if name else jp.stem
        cv2.imwrite(str(out_dir / f"{stem}.png"), mask)
        n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, required=True, help="라벨 JSON 폴더")
    p.add_argument("--images", type=Path, default=None, help="원천 이미지 폴더(크기 정보가 없을 때만 사용)")
    p.add_argument("--out", type=Path, required=True, help="마스크 PNG 저장 폴더")
    p.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--line-width", type=int, default=3)
    a = p.parse_args()
    n = convert_dir(a.labels, a.images, a.out, a.mode, a.line_width)
    print(f"마스크 {n}개 저장 -> {a.out}")


if __name__ == "__main__":
    main()

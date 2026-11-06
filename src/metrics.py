"""균열 분할 지표: 전체 픽셀 누적 IoU, Precision, Recall, F1."""

from __future__ import annotations


def batch_stats(pred, target, stats=None):
    """pred, target: bool 텐서 또는 numpy 배열(같은 모양)."""
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    if stats is None:
        stats = {"tp": 0, "fp": 0, "fn": 0}
    stats["tp"] += tp
    stats["fp"] += fp
    stats["fn"] += fn
    return stats


def summarize(stats) -> dict:
    if not stats:
        return {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "iou": tp / (tp + fp + fn + eps),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + eps),
    }

import numpy as np
import pytest

from src.augment.nuclear_conditions import CONDITIONS, apply_condition
from src.data.aihub_to_masks import parse_points, render_mask
from src.metrics import batch_stats, summarize


def label(anns, w=100, h=80):
    return {"image": {"name": "a.jpg", "width": w, "height": h, "object_included": "Y", "annotations": anns}}


def test_polyline_crack_drawn_binary():
    m, name = render_mask(label([{"label": "crack", "shape": "Polyline", "points": [[10, 10], [90, 10]]}]))
    assert name == "a.jpg" and m.shape == (80, 100)
    assert m[10, 50] == 1 and m[40, 50] == 0


def test_polygon_filled_and_non_crack_skipped_in_binary():
    poly = {"label": "spalling", "shape": "Polygon", "points": [[0, 0], [50, 0], [50, 50], [0, 50]]}
    assert render_mask(label([poly]))[0].sum() == 0
    mc, _ = render_mask(label([poly]), mode="multiclass")
    assert mc[25, 25] == 4


def test_px_sets_thickness_and_flat_points():
    thin = render_mask(label([{"label": "crack", "shape": "Polyline", "points": [10, 40, 90, 40], "px": 1}]))[0]
    thick = render_mask(label([{"label": "crack", "shape": "Polyline", "points": [10, 40, 90, 40], "px": 9}]))[0]
    assert thick.sum() > thin.sum() > 0


def test_parse_points_shapes():
    assert parse_points([[[1, 2], [3, 4]]]).shape == (2, 2)
    assert parse_points([1, 2, 3, 4]).tolist() == [[1, 2], [3, 4]]


@pytest.mark.parametrize("cond", CONDITIONS)
@pytest.mark.parametrize("level", [1, 2, 3])
def test_conditions_keep_shape_and_dtype(cond, level):
    rng = np.random.default_rng(0)
    img = np.full((64, 64, 3), 128, np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[30:34] = 1
    out, m = apply_condition(cond, img, mask, rng, level=level)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert m.shape == mask.shape and set(np.unique(m)) <= {0, 1}


def test_low_light_darkens_and_radiation_adds_hot_pixels():
    rng = np.random.default_rng(0)
    img = np.full((64, 64, 3), 128, np.uint8)
    dark, _ = apply_condition("low_light", img, np.zeros((64, 64), np.uint8), rng)
    assert dark.mean() < img.mean()
    noisy, _ = apply_condition("radiation", img, np.zeros((64, 64), np.uint8), rng)
    assert (noisy == 255).any()


def test_metrics():
    p = np.array([[1, 1, 0, 0]], bool)
    t = np.array([[1, 0, 1, 0]], bool)
    m = summarize(batch_stats(p, t))
    assert m["iou"] == pytest.approx(1 / 3) and m["f1"] == pytest.approx(0.5)

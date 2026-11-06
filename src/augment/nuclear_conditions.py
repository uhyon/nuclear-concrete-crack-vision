"""원전 현장 촬영 조건을 흉내 내는 영상 열화(degradation) 함수.

연구 질문: 일반 시설물 영상으로 학습한 균열 분할 모델이
원전 원격 점검 영상 조건에서 얼마나 성능이 떨어지고, 어떻게 회복할 수 있는가?

모든 함수는 uint8 BGR/그레이 이미지(H, W[, C])를 받아 같은 형태로 돌려준다.
마스크는 바뀌지 않는다(기하 변형인 narrow_fov는 마스크도 함께 변환).

주의: 이 함수들은 실제 방사선 영상 노이즈의 근사치다. 실제 고방사선 환경 영상과의
비교 검증은 3단계 과제로 남긴다.
"""

from __future__ import annotations

import cv2
import numpy as np


def radiation_noise(
    img: np.ndarray,
    rng: np.random.Generator,
    spot_density: float = 0.002,
    streak_count: int = 8,
    gaussian_sigma: float = 6.0,
) -> np.ndarray:
    """감마선이 이미지 센서에 닿을 때 생기는 흰 점(hot pixel)과 짧은 줄, 전체 잡음."""
    out = img.astype(np.float32)
    h, w = img.shape[:2]

    # 전체 가우시안 잡음
    out += rng.normal(0.0, gaussian_sigma, size=out.shape)

    # 흰 점: 1~2픽셀 크기의 포화 점
    n_spots = int(h * w * spot_density)
    ys = rng.integers(0, h, n_spots)
    xs = rng.integers(0, w, n_spots)
    out[ys, xs] = 255.0
    big = rng.random(n_spots) < 0.2
    ys2 = np.clip(ys[big] + 1, 0, h - 1)
    xs2 = np.clip(xs[big] + 1, 0, w - 1)
    out[ys2, xs[big]] = 255.0
    out[ys[big], xs2] = 255.0

    # 짧은 밝은 줄(입자 궤적)
    for _ in range(streak_count):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        length = int(rng.integers(5, max(6, min(h, w) // 8)))
        angle = rng.uniform(0, np.pi)
        x1 = int(np.clip(x0 + length * np.cos(angle), 0, w - 1))
        y1 = int(np.clip(y0 + length * np.sin(angle), 0, h - 1))
        color = (255.0,) * (out.shape[2] if out.ndim == 3 else 1)
        cv2.line(out, (x0, y0), (x1, y1), color, thickness=1)

    return np.clip(out, 0, 255).astype(np.uint8)


def low_light(
    img: np.ndarray,
    rng: np.random.Generator,
    brightness: float = 0.35,
    gamma: float = 1.6,
    read_noise_sigma: float = 4.0,
) -> np.ndarray:
    """조명이 부족한 밀폐 공간: 어둡게 + 감마 + 센서 읽기 잡음."""
    x = img.astype(np.float32) / 255.0
    x = np.power(x, gamma) * brightness
    x = x * 255.0 + rng.normal(0.0, read_noise_sigma, size=x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def uneven_light(img: np.ndarray, rng: np.random.Generator, strength: float = 0.6) -> np.ndarray:
    """로봇 탑재 조명 하나로 비출 때 생기는 중심 밝고 가장자리 어두운 패턴."""
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.hypot(w, h)
    gain = 1.0 - strength * np.clip(d * 2.0, 0, 1)
    if img.ndim == 3:
        gain = gain[..., None]
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def motion_blur(img: np.ndarray, rng: np.random.Generator, max_kernel: int = 9) -> np.ndarray:
    """이동 중인 로봇·드론 촬영의 흔들림."""
    k = int(rng.integers(3, max_kernel + 1)) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    angle = rng.uniform(0, 180)
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def narrow_fov(
    img: np.ndarray,
    mask: np.ndarray | None,
    rng: np.random.Generator,
    zoom: float = 1.8,
    distortion: float = 0.25,
) -> tuple[np.ndarray, np.ndarray | None]:
    """내시경·소형 로봇 카메라: 좁은 시야(확대 크롭) + 술통형 왜곡 + 비네팅."""
    h, w = img.shape[:2]
    ch, cw = int(h / zoom), int(w / zoom)
    y0 = int(rng.integers(0, h - ch + 1))
    x0 = int(rng.integers(0, w - cw + 1))
    img_c = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    mask_c = None
    if mask is not None:
        mask_c = cv2.resize(mask[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_NEAREST)

    # 술통형 왜곡
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
    r2 = nx ** 2 + ny ** 2
    factor = 1 + distortion * r2
    map_x = (nx * factor) * (w / 2) + w / 2
    map_y = (ny * factor) * (h / 2) + h / 2
    img_d = cv2.remap(img_c, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if mask_c is not None:
        mask_c = cv2.remap(mask_c, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    # 비네팅
    vign = np.clip(1.0 - 0.5 * r2, 0.2, 1.0)
    if img_d.ndim == 3:
        vign = vign[..., None]
    img_d = np.clip(img_d.astype(np.float32) * vign, 0, 255).astype(np.uint8)
    return img_d, mask_c


# 평가에 쓰는 조건 묶음. 강도(level)는 1(약)~3(강).
def apply_condition(
    name: str,
    img: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    level: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    s = {1: 0.5, 2: 1.0, 3: 2.0}[level]
    if name == "clean":
        return img, mask
    if name == "radiation":
        return radiation_noise(img, rng, spot_density=0.002 * s, streak_count=int(8 * s), gaussian_sigma=6 * s), mask
    if name == "low_light":
        return low_light(img, rng, brightness=max(0.1, 0.5 / s), gamma=1 + 0.6 * s, read_noise_sigma=4 * s), mask
    if name == "uneven_light":
        return uneven_light(img, rng, strength=min(0.9, 0.4 * s)), mask
    if name == "motion_blur":
        return motion_blur(img, rng, max_kernel=int(5 + 4 * s)), mask
    if name == "narrow_fov":
        out, m = narrow_fov(img, mask, rng, zoom=1 + 0.5 * s, distortion=0.15 * s)
        return out, m
    if name == "combined":
        out, m = narrow_fov(img, mask, rng, zoom=1 + 0.4 * s, distortion=0.1 * s)
        out = low_light(out, rng, brightness=max(0.15, 0.6 / s), gamma=1 + 0.4 * s, read_noise_sigma=3 * s)
        out = radiation_noise(out, rng, spot_density=0.0015 * s, streak_count=int(6 * s), gaussian_sigma=4 * s)
        return out, m
    raise ValueError(f"알 수 없는 조건: {name}")


CONDITIONS = ["clean", "radiation", "low_light", "uneven_light", "motion_blur", "narrow_fov", "combined"]

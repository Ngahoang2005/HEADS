#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
segment_v4_pixel.py

Word-level trainable segment grouping for FUNSD with pixel evidence.

Main idea
---------
ONE LayoutLMv3 forward per page
        |
        v
word hidden states
        |
        v
symmetric word-pair relation head
        |
        v
P(same FUNSD form.id)
        |
        v
adaptive graph clustering
        |
        v
final layout segments

Important
---------
Standard FUNSD annotations do NOT contain segment_id.

Therefore the default gold target is:

    same_segment(i, j) = (form.id_i == form.id_j)

The "linking" field is NOT treated as a segment label.

This version intentionally removes the old:

    word -> row -> run -> mean hidden

pipeline.

Reason:
------
The previous pipeline created many mixed runs:

    "TO: George Baroody"

could contain:

    TO       -> form 1
    George   -> form 14
    Baroody  -> form 14

which makes a single averaged run representation ambiguous.

This version keeps each word separate.

V4 additions
------------
This version adds image-pixel evidence explicitly. The same page image is
blurred at multiple scales to estimate local ink density, corridor continuity,
and gap/separator evidence between word boxes. These features are fed to the
relation head and used as a conservative clustering tie-breaker / geometry
relaxation. LayoutLMv3 remains frozen and still supplies the multimodal word
hidden states.

Compatible with:
    Python 3.7
    torch 1.10
    transformers 4.16.2
    local layoutlmft LayoutLMv3

Default repository:
    /home/s24gbn1/Documents/httn/unilm/layoutlmv3

Commands
--------

1) TRAIN

python segment_v4_pixel.py train \
    --train-images datasets/funsd/dataset/training_data/images \
    --train-annotations datasets/funsd/dataset/training_data/annotations \
    --epochs 15 \
    --head-out segment_outputs/segment_head_v4_pixel.pt

2) DIAGNOSE RAW MLP ON TEST

python segment_v4_pixel.py diagnose \
    --head segment_outputs/segment_head_v4_pixel.pt \
    --images datasets/funsd/dataset/testing_data/images \
    --annotations datasets/funsd/dataset/testing_data/annotations \
    --output segment_outputs/diagnostic_v4_pixel

3) PREDICT / EVALUATE TEST

python segment_v4_pixel.py predict \
    --head segment_outputs/segment_head_v4_pixel.pt \
    --images datasets/funsd/dataset/testing_data/images \
    --annotations datasets/funsd/dataset/testing_data/annotations \
    --evaluate \
    --output segment_outputs/test_v4_pixel
"""

from __future__ import print_function

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageFilter

import numpy as np


# =============================================================================
# Constants
# =============================================================================

REPO = Path(__file__).resolve().parent

DEFAULT_MODEL = str(REPO / "models/layoutlmv3-base")

DEFAULT_TRAIN_IMAGES = str(
    REPO / "datasets/funsd/dataset/training_data/images"
)

DEFAULT_TRAIN_ANN = str(
    REPO / "datasets/funsd/dataset/training_data/annotations"
)

DEFAULT_TEST_IMAGES = str(
    REPO / "datasets/funsd/dataset/testing_data/images"
)

DEFAULT_TEST_ANN = str(
    REPO / "datasets/funsd/dataset/testing_data/annotations"
)

DEFAULT_OUT = str(REPO / "segment_outputs")

MAX_TEXT_LEN = 512

VISUAL_TOKENS = 197

HIDDEN_PROJECTION = 192

GEOMETRY_DIM = 20

PIXEL_DIM = 16

PIXEL_MAX_DIM = 512

DEFAULT_TOP_K = 32


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def h(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y0 + self.y1)

    def union(self, other: "Box") -> "Box":
        return Box(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )


@dataclass
class Word:
    idx: int
    text: str
    box: Box
    segment_id: Optional[str] = None
    entity_id: int = -1
    entity_label: str = "other"
    token_positions: Optional[List[int]] = None
    hidden: Any = None
    row_id: int = -1


@dataclass
class RelationHeadConfig:
    hidden_size: int
    projection_size: int = HIDDEN_PROJECTION
    geometry_dim: int = GEOMETRY_DIM
    pixel_dim: int = PIXEL_DIM
    width: int = 256
    bottleneck: int = 64
    dropout: float = 0.10


# =============================================================================
# Utilities
# =============================================================================

def safe_div(a, b):
    if b == 0:
        return 0.0
    return float(a) / float(b)


def median(xs, default=10.0):
    values = [
        float(x)
        for x in xs
        if x is not None and math.isfinite(float(x))
    ]

    if not values:
        return default

    return float(statistics.median(values))


def percentile(values, q):
    if not values:
        return 0.0

    xs = sorted(float(v) for v in values)

    if len(xs) == 1:
        return xs[0]

    q = max(0.0, min(1.0, float(q)))

    pos = q * (len(xs) - 1)

    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return xs[lo]

    alpha = pos - lo

    return xs[lo] * (1.0 - alpha) + xs[hi] * alpha


def union_boxes(boxes):
    boxes = list(boxes)

    if not boxes:
        return Box(0, 0, 0, 0)

    out = boxes[0]

    for b in boxes[1:]:
        out = out.union(b)

    return out


def parse_box(raw):
    x0, y0, x1, y1 = [
        float(v)
        for v in raw
    ]

    return Box(
        min(x0, x1),
        min(y0, y1),
        max(x0, x1),
        max(y0, y1),
    )


def normalize_1000(box, page_w, page_h):
    def q(v, d):
        return int(
            max(
                0,
                min(
                    1000,
                    round(
                        1000.0 * v / max(1.0, d)
                    )
                )
            )
        )

    return [
        q(box.x0, page_w),
        q(box.y0, page_h),
        q(box.x1, page_w),
        q(box.y1, page_h),
    ]


def gap_x(a, b):
    if a.x1 < b.x0:
        return b.x0 - a.x1

    if b.x1 < a.x0:
        return a.x0 - b.x1

    return 0.0


def gap_y(a, b):
    if a.y1 < b.y0:
        return b.y0 - a.y1

    if b.y1 < a.y0:
        return a.y0 - b.y1

    return 0.0


def x_overlap(a, b):
    inter = max(
        0.0,
        min(a.x1, b.x1) -
        max(a.x0, b.x0)
    )

    return inter / max(
        1e-6,
        min(a.w, b.w)
    )


def y_overlap(a, b):
    inter = max(
        0.0,
        min(a.y1, b.y1) -
        max(a.y0, b.y0)
    )

    return inter / max(
        1e-6,
        min(a.h, b.h)
    )


# =============================================================================
# Pixel / ink evidence
# =============================================================================

def otsu_threshold(gray01):

    values = np.clip(
        (gray01 * 255.0).astype(np.uint8),
        0,
        255,
    )

    hist = np.bincount(
        values.reshape(-1),
        minlength=256,
    ).astype(np.float64)

    total = values.size

    if total <= 0:
        return 0.70

    weighted_total = float(
        np.dot(
            np.arange(256),
            hist,
        )
    )

    sum_back = 0.0
    weight_back = 0.0
    best = 0.0
    best_t = 128

    for t in range(256):

        weight_back += hist[t]

        if weight_back <= 0:
            continue

        weight_fore = total - weight_back

        if weight_fore <= 0:
            break

        sum_back += float(t * hist[t])

        mean_back = sum_back / weight_back
        mean_fore = (
            weighted_total - sum_back
        ) / weight_fore

        between = (
            weight_back
            * weight_fore
            * (mean_back - mean_fore) ** 2
        )

        if between > best:
            best = between
            best_t = t

    # FUNSD scans are usually bright-background / dark-ink. Keep a sane
    # threshold even when the page contains a stamp or a very large blank area.
    threshold = max(
        0.35,
        min(
            0.90,
            best_t / 255.0,
        ),
    )

    return float(threshold)


def resize_for_pixels(image, max_dim=PIXEL_MAX_DIM):

    w, h = image.size

    if max(w, h) <= max_dim:
        return image.copy()

    scale = float(max_dim) / float(max(w, h))

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    return image.resize(
        (nw, nh),
        Image.LANCZOS,
    )


def integral_image(arr):

    arr = np.asarray(
        arr,
        dtype=np.float32,
    )

    return np.pad(
        arr.cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0.0,
    )


def rect_mean(ii, x0, y0, x1, y1):

    h = ii.shape[0] - 1
    w = ii.shape[1] - 1

    x0 = max(0, min(w, int(math.floor(x0))))
    y0 = max(0, min(h, int(math.floor(y0))))
    x1 = max(x0 + 1, min(w, int(math.ceil(x1))))
    y1 = max(y0 + 1, min(h, int(math.ceil(y1))))

    area = float(
        max(1, x1 - x0)
        * max(1, y1 - y0)
    )

    total = (
        ii[y1, x1]
        - ii[y0, x1]
        - ii[y1, x0]
        + ii[y0, x0]
    )

    return float(total / area)


def box_to_pixel(box, page_box, width, height):

    sx = float(width) / max(page_box.w, 1.0)
    sy = float(height) / max(page_box.h, 1.0)

    return (
        box.x0 * sx,
        box.y0 * sy,
        box.x1 * sx,
        box.y1 * sy,
    )


def build_pixel_context(image, page_box):

    small = resize_for_pixels(
        image.convert("L"),
        PIXEL_MAX_DIM,
    )

    gray = np.asarray(
        small,
        dtype=np.float32,
    ) / 255.0

    threshold = otsu_threshold(gray)

    # Dark-pixel / ink mask.
    ink = (
        gray < threshold
    ).astype(np.float32)

    # Very small isolated noise is less useful than contiguous ink. We do not
    # run a hard connected-component algorithm because FUNSD contains stamps,
    # underlines and border noise. Instead, multi-scale blur acts as the
    # requested "text sticks together" signal.
    mask_img = Image.fromarray(
        (ink * 255.0).astype(np.uint8),
        mode="L",
    )

    blur_arrays = []

    for radius in (2.0, 6.0, 14.0):

        blurred = mask_img.filter(
            ImageFilter.GaussianBlur(
                radius=radius
            )
        )

        blur_arrays.append(
            np.asarray(
                blurred,
                dtype=np.float32,
            ) / 255.0
        )

    pixel_page_box = Box(
        0.0,
        0.0,
        float(image.size[0]),
        float(image.size[1]),
    )

    return {
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "threshold": threshold,
        "gray": gray,
        "ink": ink,
        "ink_integral": integral_image(ink),
        "gray_integral": integral_image(gray),
        "blur": blur_arrays,
        "blur_integral": [
            integral_image(x)
            for x in blur_arrays
        ],
        "page_box": pixel_page_box,
    }


def word_pixel_features(word, context, med_h):

    page_box = context["page_box"]
    width = context["width"]
    height = context["height"]

    x0, y0, x1, y1 = box_to_pixel(
        word.box,
        page_box,
        width,
        height,
    )

    # Slightly enlarged local window captures nearby whitespace/ink context.
    pad_x = 0.35 * max(
        x1 - x0,
        1.0,
    )
    pad_y = 0.50 * max(
        y1 - y0,
        1.0,
    )

    ex0 = x0 - pad_x
    ey0 = y0 - pad_y
    ex1 = x1 + pad_x
    ey1 = y1 + pad_y

    local_ink = rect_mean(
        context["ink_integral"],
        x0,
        y0,
        x1,
        y1,
    )

    expanded_ink = rect_mean(
        context["ink_integral"],
        ex0,
        ey0,
        ex1,
        ey1,
    )

    gray_mean = rect_mean(
        context["gray_integral"],
        x0,
        y0,
        x1,
        y1,
    )

    b1 = rect_mean(
        context["blur_integral"][0],
        ex0,
        ey0,
        ex1,
        ey1,
    )

    b2 = rect_mean(
        context["blur_integral"][1],
        ex0,
        ey0,
        ex1,
        ey1,
    )

    b3 = rect_mean(
        context["blur_integral"][2],
        ex0,
        ey0,
        ex1,
        ey1,
    )

    return {
        "local_ink": local_ink,
        "expanded_ink": expanded_ink,
        "gray_mean": gray_mean,
        "blur1": b1,
        "blur2": b2,
        "blur3": b3,
    }


def pixel_pair_features(a, b, context, word_pixel_a, word_pixel_b, med_h):

    page_box = context["page_box"]
    width = context["width"]
    height = context["height"]

    ax0, ay0, ax1, ay1 = box_to_pixel(
        a.box,
        page_box,
        width,
        height,
    )

    bx0, by0, bx1, by1 = box_to_pixel(
        b.box,
        page_box,
        width,
        height,
    )

    acx = 0.5 * (ax0 + ax1)
    acy = 0.5 * (ay0 + ay1)
    bcx = 0.5 * (bx0 + bx1)
    bcy = 0.5 * (by0 + by1)

    dx = abs(bcx - acx)
    dy = abs(bcy - acy)

    # A thin corridor between the two word centers. This is the key
    # multi-scale "blur/bridge" signal: if text lies on a continuous region,
    # medium/large blur tends to remain non-zero through the corridor.
    if dx >= dy:

        half_t = max(
            1.0,
            0.70 * med_h * width / max(page_box.w, 1.0),
            0.25 * max(
                ay1 - ay0,
                by1 - by0,
            ),
        )

        cx0 = min(acx, bcx)
        cx1 = max(acx, bcx)

        cy0 = 0.5 * (acy + bcy) - half_t
        cy1 = 0.5 * (acy + bcy) + half_t

        h_corr_ink = rect_mean(
            context["ink_integral"],
            cx0,
            cy0,
            cx1 + 1.0,
            cy1,
        )

        h_corr_blur = rect_mean(
            context["blur_integral"][2],
            cx0,
            cy0,
            cx1 + 1.0,
            cy1,
        )

        h_corr = h_corr_blur
        v_corr_ink = 0.0
        v_corr_blur = 0.0
        horizontality = 1.0

    else:

        half_t = max(
            1.0,
            0.70 * med_h * height / max(page_box.h, 1.0),
            0.25 * max(
                ax1 - ax0,
                bx1 - bx0,
            ),
        )

        cy0 = min(acy, bcy)
        cy1 = max(acy, bcy)

        cx0 = 0.5 * (acx + bcx) - half_t
        cx1 = 0.5 * (acx + bcx) + half_t

        v_corr_ink = rect_mean(
            context["ink_integral"],
            cx0,
            cy0,
            cx1,
            cy1 + 1.0,
        )

        v_corr_blur = rect_mean(
            context["blur_integral"][2],
            cx0,
            cy0,
            cx1,
            cy1 + 1.0,
        )

        h_corr = v_corr_blur
        h_corr_ink = 0.0
        h_corr_blur = 0.0
        horizontality = 0.0

    # Gap-strip signal. We explicitly measure the actual whitespace/ink gap
    # between boxes, which helps penalize merges across unrelated regions.
    gx = gap_x(a.box, b.box)
    gy = gap_y(a.box, b.box)

    gap_ink = 0.0
    gap_blur = 0.0

    if gx > 0 and y_overlap(a.box, b.box) >= 0.05:

        qx0 = min(ax1, bx1)
        qx1 = max(ax0, bx0)
        qy0 = min(ay0, by0)
        qy1 = max(ay1, by1)

        gap_ink = rect_mean(
            context["ink_integral"],
            qx0,
            qy0,
            qx1,
            qy1,
        )

        gap_blur = rect_mean(
            context["blur_integral"][2],
            qx0,
            qy0,
            qx1,
            qy1,
        )

    elif gy > 0 and x_overlap(a.box, b.box) >= 0.05:

        qx0 = min(ax0, bx0)
        qx1 = max(ax1, bx1)
        qy0 = min(ay1, by1)
        qy1 = max(ay0, by0)

        gap_ink = rect_mean(
            context["ink_integral"],
            qx0,
            qy0,
            qx1,
            qy1,
        )

        gap_blur = rect_mean(
            context["blur_integral"][2],
            qx0,
            qy0,
            qx1,
            qy1,
        )

    ux0 = min(ax0, bx0)
    uy0 = min(ay0, by0)
    ux1 = max(ax1, bx1)
    uy1 = max(ay1, by1)

    union_blur = rect_mean(
        context["blur_integral"][2],
        ux0,
        uy0,
        ux1,
        uy1,
    )

    min_blur3 = min(
        word_pixel_a["blur3"],
        word_pixel_b["blur3"],
    )

    max_local_blur3 = max(
        word_pixel_a["blur3"],
        word_pixel_b["blur3"],
    )

    # 16 fixed-dim pixel features, all kept on a modest 0..1-ish scale.
    return [
        min(
            word_pixel_a["local_ink"],
            word_pixel_b["local_ink"],
        ),
        max(
            word_pixel_a["local_ink"],
            word_pixel_b["local_ink"],
        ),
        abs(
            word_pixel_a["local_ink"]
            - word_pixel_b["local_ink"]
        ),
        min(
            word_pixel_a["blur1"],
            word_pixel_b["blur1"],
        ),
        min(
            word_pixel_a["blur2"],
            word_pixel_b["blur2"],
        ),
        min_blur3,
        0.5 * (
            word_pixel_a["blur3"]
            + word_pixel_b["blur3"]
        ),
        min(
            1.0,
            h_corr_ink,
        ),
        min(
            1.0,
            v_corr_ink,
        ),
        min(
            1.0,
            h_corr_blur,
        ),
        min(
            1.0,
            v_corr_blur,
        ),
        min(
            1.0,
            gap_ink,
        ),
        min(
            1.0,
            gap_blur,
        ),
        min(
            1.0,
            union_blur,
        ),
        max(
            -1.0,
            min(
                1.0,
                max_local_blur3 - gap_blur,
            ),
        ),
        horizontality,
    ]


def pixel_link_score(feature):

    if feature is None or len(feature) < PIXEL_DIM:
        return 0.0

    # Use only positive bridge/context signals. Gap ink is a mild penalty.
    score = (
        0.18 * feature[3]
        + 0.20 * feature[4]
        + 0.20 * feature[5]
        + 0.16 * max(
            feature[9],
            feature[10],
        )
        + 0.16 * feature[13]
        + 0.10 * max(
            0.0,
            feature[14],
        )
        - 0.08 * feature[12]
    )

    return float(
        max(
            0.0,
            min(
                1.0,
                score,
            ),
        )
    )


def make_pair_pixel_features(
    words,
    pairs,
    context,
):

    import torch

    if not pairs:
        return torch.zeros(
            (0, PIXEL_DIM),
            dtype=torch.float32,
        )

    med_h = median(
        [
            w.box.h
            for w in words
            if w.box.h > 0
        ],
        10.0,
    )

    cache = []

    for w in words:
        cache.append(
            word_pixel_features(
                w,
                context,
                med_h,
            )
        )

    values = []

    for i, j in pairs:
        values.append(
            pixel_pair_features(
                words[i],
                words[j],
                context,
                cache[i],
                cache[j],
                med_h,
            )
        )

    return torch.tensor(
        values,
        dtype=torch.float32,
    )


def quick_pixel_link_score(i, j, words, context, word_cache, med_h):

    feature = pixel_pair_features(
        words[i],
        words[j],
        context,
        word_cache[i],
        word_cache[j],
        med_h,
    )

    return pixel_link_score(feature)


# =============================================================================
# LayoutLMv3
# =============================================================================

def ensure_repo_importable():
    repo = Path(__file__).resolve().parent

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    return repo


def load_layoutlmv3(model_path, device):

    import torch
    import transformers

    ensure_repo_importable()

    from layoutlmft.models.layoutlmv3 import (
        LayoutLMv3Model,
        LayoutLMv3TokenizerFast,
    )

    print(
        "[MODEL] transformers={}".format(
            transformers.__version__
        )
    )

    print(
        "[MODEL] implementation={}.{}".format(
            LayoutLMv3Model.__module__,
            LayoutLMv3Model.__name__,
        )
    )

    print(
        "[MODEL] tokenizer={}.{}".format(
            LayoutLMv3TokenizerFast.__module__,
            LayoutLMv3TokenizerFast.__name__,
        )
    )

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    required = [
        "config.json",
        "pytorch_model.bin",
        "vocab.json",
        "merges.txt",
    ]

    missing = [
        x
        for x in required
        if not (model_path / x).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Incomplete LayoutLMv3 checkpoint {}: {}".format(
                model_path,
                missing,
            )
        )

    try:
        tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
            str(model_path),
            local_files_only=True,
            add_prefix_space=True,
        )
    except TypeError:
        tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
            str(model_path),
            local_files_only=True,
        )

    model = LayoutLMv3Model.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    model.eval()

    model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    print(
        "[MODEL] backbone frozen"
    )

    print(
        "[MODEL] one LayoutLMv3 forward per page"
    )

    return tokenizer, model


def build_image_transform():

    from torchvision import transforms

    try:
        from timm.data.constants import (
            IMAGENET_INCEPTION_MEAN,
            IMAGENET_INCEPTION_STD,
        )

        mean = IMAGENET_INCEPTION_MEAN
        std = IMAGENET_INCEPTION_STD

    except Exception:
        mean = (
            0.485,
            0.456,
            0.406,
        )

        std = (
            0.229,
            0.224,
            0.225,
        )

    return transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=3,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )


def build_inputs(
    tokenizer,
    image,
    words,
    page_box,
    max_length=MAX_TEXT_LEN,
):

    import torch

    texts = [
        w.text
        for w in words
    ]

    bboxes = [
        normalize_1000(
            w.box,
            page_box.w,
            page_box.h,
        )
        for w in words
    ]

    enc = tokenizer(
        texts,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    word_ids = enc.word_ids(
        batch_index=0
    )

    bbox_inputs = []

    for wid in word_ids:

        if wid is None:
            bbox_inputs.append(
                [0, 0, 0, 0]
            )

        elif wid < 0 or wid >= len(bboxes):
            bbox_inputs.append(
                [0, 0, 0, 0]
            )

        else:
            bbox_inputs.append(
                bboxes[wid]
            )

    enc["bbox"] = torch.tensor(
        [bbox_inputs],
        dtype=torch.long,
    )

    enc["token_type_ids"] = torch.zeros_like(
        enc["input_ids"],
        dtype=torch.long,
    )

    transform = build_image_transform()

    enc["images"] = transform(
        image.convert("RGB")
    ).unsqueeze(0)

    text_mask = enc["attention_mask"]

    visual_mask = torch.ones(
        (
            text_mask.shape[0],
            VISUAL_TOKENS,
        ),
        dtype=text_mask.dtype,
    )

    enc["attention_mask"] = torch.cat(
        [
            text_mask,
            visual_mask,
        ],
        dim=1,
    )

    return enc, word_ids


def extract_word_hidden(
    tokenizer,
    model,
    image,
    words,
    page_box,
    device,
    max_length,
):

    import torch

    enc, word_ids = build_inputs(
        tokenizer,
        image,
        words,
        page_box,
        max_length=max_length,
    )

    text_len = int(
        enc["input_ids"].shape[1]
    )

    used_word_ids = [
        wid
        for wid in word_ids
        if wid is not None
    ]

    covered = len(
        set(used_word_ids)
    )

    if covered < len(words):

        print(
            "[WARN] tokenizer covered {}/{} words".format(
                covered,
                len(words),
            )
        )

    tensors = {}

    for k, v in enc.items():
        if hasattr(v, "to"):
            tensors[k] = v.to(device)
        else:
            tensors[k] = v

    print(
        "[INPUT] text_tokens={} visual_tokens={} "
        "encoder_tokens={} attention_mask={}".format(
            text_len,
            VISUAL_TOKENS,
            text_len + VISUAL_TOKENS,
            tensors["attention_mask"].shape[1],
        )
    )

    with torch.no_grad():

        out = model(
            **tensors,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

    hidden = (
        out.last_hidden_state[0]
        .detach()
        .float()
        .cpu()
    )

    token_positions = defaultdict(list)

    for pos, wid in enumerate(word_ids):

        if wid is not None and 0 <= wid < len(words):

            token_positions[
                int(wid)
            ].append(pos)

    zero_count = 0

    for w in words:

        positions = token_positions.get(
            w.idx,
            [],
        )

        w.token_positions = list(
            positions
        )

        if positions:

            w.hidden = hidden[
                positions
            ].mean(
                dim=0
            )

        else:

            w.hidden = torch.zeros(
                hidden.shape[-1]
            )

            zero_count += 1

    if zero_count > 0:

        print(
            "[WARN] {} words have no token representation".format(
                zero_count
            )
        )

    return covered


# =============================================================================
# FUNSD
# =============================================================================

def flatten_funsd_words(data):

    words = []

    entities = {}

    next_word = 0

    for form in data.get(
        "form",
        [],
    ):

        try:
            eid = int(
                form.get(
                    "id",
                    -1,
                )
            )
        except Exception:
            eid = -1

        label = str(
            form.get(
                "label",
                "other",
            )
        )

        entities[eid] = {
            "label": label,
            "text": form.get(
                "text",
                "",
            ),
        }

        for raw_word in form.get(
            "words",
            [],
        ):

            text = str(
                raw_word.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            box = parse_box(
                raw_word.get(
                    "box",
                    [
                        0,
                        0,
                        0,
                        0,
                    ],
                )
            )

            words.append(
                Word(
                    idx=next_word,
                    text=text,
                    box=box,
                    entity_id=eid,
                    entity_label=label,
                )
            )

            next_word += 1

    return words, entities


def load_page_annotation(path):

    with open(
        str(path),
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    words, entities = flatten_funsd_words(
        data
    )

    # Standard FUNSD:
    # form.id is the gold grouping target.
    for w in words:

        if w.entity_id >= 0:
            w.segment_id = str(
                w.entity_id
            )
        else:
            w.segment_id = None

    return words, entities


# =============================================================================
# Row estimation
#
# IMPORTANT:
# This does NOT merge words into runs.
# It only gives geometry features a row_id.
# =============================================================================

def assign_word_rows(words):

    if not words:
        return

    order = sorted(
        range(len(words)),
        key=lambda i: (
            words[i].box.cy,
            words[i].box.x0,
        ),
    )

    med_h = median(
        [
            words[i].box.h
            for i in order
            if words[i].box.h > 0
        ],
        10.0,
    )

    rows = []

    for wid in order:

        w = words[wid]

        best_row = None
        best_score = -1e9

        for rid, ids in enumerate(rows):

            rb = union_boxes(
                [
                    words[j].box
                    for j in ids
                ]
            )

            yo = y_overlap(
                w.box,
                rb,
            )

            center_dist = abs(
                w.box.cy - rb.cy
            ) / max(
                med_h,
                1.0,
            )

            score = (
                yo
                - 0.20 * center_dist
            )

            if (
                yo >= 0.10
                and center_dist <= 0.90
                and score > best_score
            ):

                best_score = score
                best_row = rid

        if best_row is None:

            rows.append(
                [wid]
            )

        else:

            rows[
                best_row
            ].append(
                wid
            )

    for rid, ids in enumerate(rows):

        for wid in ids:
            words[wid].row_id = rid


# =============================================================================
# Word-level geometry
# =============================================================================

def geometry_features(
    a,
    b,
    page_box,
    med_h,
):

    dx = (
        abs(
            b.box.cx - a.box.cx
        )
        / max(
            page_box.w,
            1.0,
        )
    )

    dy = (
        abs(
            b.box.cy - a.box.cy
        )
        / max(
            page_box.h,
            1.0,
        )
    )

    gx = (
        gap_x(
            a.box,
            b.box,
        )
        / max(
            page_box.w,
            1.0,
        )
    )

    gy = (
        gap_y(
            a.box,
            b.box,
        )
        / max(
            page_box.h,
            1.0,
        )
    )

    xo = x_overlap(
        a.box,
        b.box,
    )

    yo = y_overlap(
        a.box,
        b.box,
    )

    same_row = (
        1.0
        if a.row_id == b.row_id
        else 0.0
    )

    row_delta = (
        abs(
            a.row_id - b.row_id
        )
        / 20.0
    )

    width_ratio = (
        min(
            a.box.w,
            b.box.w,
        )
        /
        max(
            max(
                a.box.w,
                b.box.w,
            ),
            1.0,
        )
    )

    height_ratio = (
        min(
            a.box.h,
            b.box.h,
        )
        /
        max(
            max(
                a.box.h,
                b.box.h,
            ),
            1.0,
        )
    )

    vertical_distance_h = (
        gap_y(
            a.box,
            b.box,
        )
        /
        max(
            med_h,
            1.0,
        )
    )

    horizontal_distance_h = (
        gap_x(
            a.box,
            b.box,
        )
        /
        max(
            med_h,
            1.0,
        )
    )

    near_vertical = (
        1.0
        if vertical_distance_h <= 3.0
        else 0.0
    )

    near_vertical_8 = (
        1.0
        if vertical_distance_h <= 8.0
        else 0.0
    )

    near_horizontal = (
        1.0
        if horizontal_distance_h <= 3.0
        else 0.0
    )

    near_horizontal_8 = (
        1.0
        if horizontal_distance_h <= 8.0
        else 0.0
    )

    aligned_x = (
        1.0
        if (
            xo > 0.10
            or abs(
                a.box.cx - b.box.cx
            )
            <= 0.50 * max(
                a.box.w,
                b.box.w,
            )
        )
        else 0.0
    )

    aligned_y = (
        1.0
        if (
            yo > 0.10
            or abs(
                a.box.cy - b.box.cy
            )
            <= 0.50 * max(
                a.box.h,
                b.box.h,
            )
        )
        else 0.0
    )

    return [
        dx,
        dy,
        gx,
        gy,
        xo,
        yo,
        same_row,
        row_delta,
        width_ratio,
        height_ratio,
        vertical_distance_h / 20.0,
        horizontal_distance_h / 20.0,
        near_vertical,
        near_vertical_8,
        near_horizontal,
        near_horizontal_8,
        aligned_x,
        aligned_y,
        min(
            dx,
            1.0,
        ),
        min(
            dy,
            1.0,
        ),
    ]


# =============================================================================
# Candidate generation
#
# Goal:
#   high recall, NOT aggressive filtering.
#
# We use:
#   1. all pairs for small pages
#   2. nearest geometric neighbors
#   3. reading-order neighbors
#   4. same-row neighbors
#   5. nearby vertical neighbors
#
# This is intentionally much broader than the old candidate_pairs().
# =============================================================================

def candidate_pairs(
    words,
    page_box,
    top_k=DEFAULT_TOP_K,
    all_pair_limit=180,
    pixel_context=None,
    pixel_top_k=24,
):

    n = len(words)

    if n <= 1:
        return []

    assign_word_rows(
        words
    )

    med_h = median(
        [
            w.box.h
            for w in words
            if w.box.h > 0
        ],
        10.0,
    )

    pairs = set()

    # -------------------------------------------------------------------------
    # Small page -> all pairs
    # -------------------------------------------------------------------------

    if n <= all_pair_limit:

        for i in range(n):

            for j in range(
                i + 1,
                n,
            ):

                pairs.add(
                    (i, j)
                )

        return sorted(
            pairs
        )

    # -------------------------------------------------------------------------
    # 1) Reading-order neighbors
    # -------------------------------------------------------------------------

    reading = sorted(
        range(n),
        key=lambda i: (
            words[i].box.cy,
            words[i].box.x0,
        ),
    )

    reading_window = max(
        16,
        min(
            48,
            top_k,
        )
    )

    for pos in range(n):

        i = reading[pos]

        for delta in range(
            1,
            reading_window + 1,
        ):

            jpos = pos + delta

            if jpos >= n:
                break

            j = reading[jpos]

            if i != j:

                if i < j:
                    pairs.add(
                        (i, j)
                    )
                else:
                    pairs.add(
                        (j, i)
                    )

    # -------------------------------------------------------------------------
    # 2) Same-row pairs
    # -------------------------------------------------------------------------

    rows = defaultdict(list)

    for i, w in enumerate(words):

        rows[
            w.row_id
        ].append(i)

    for rid, ids in rows.items():

        ids.sort(
            key=lambda i:
            words[i].box.x0
        )

        # Avoid quadratic explosion inside extremely long rows.
        local_k = min(
            24,
            top_k,
        )

        for pos, i in enumerate(ids):

            for delta in range(
                1,
                local_k + 1,
            ):

                if pos + delta >= len(ids):
                    break

                j = ids[
                    pos + delta
                ]

                if i < j:
                    pairs.add(
                        (i, j)
                    )
                else:
                    pairs.add(
                        (j, i)
                    )

    # -------------------------------------------------------------------------
    # 3) Geometric nearest neighbors
    # -------------------------------------------------------------------------

    for i, a in enumerate(words):

        candidates = []

        for j, b in enumerate(words):

            if i == j:
                continue

            dx = (
                abs(
                    a.box.cx - b.box.cx
                )
                /
                max(
                    page_box.w,
                    1.0,
                )
            )

            dy = (
                abs(
                    a.box.cy - b.box.cy
                )
                /
                max(
                    page_box.h,
                    1.0,
                )
            )

            gx = (
                gap_x(
                    a.box,
                    b.box,
                )
                /
                max(
                    page_box.w,
                    1.0,
                )
            )

            gy = (
                gap_y(
                    a.box,
                    b.box,
                )
                /
                max(
                    page_box.h,
                    1.0,
                )
            )

            # Geometry distance.
            dist = (
                1.0 * dx
                + 1.25 * dy
                + 0.75 * gx
                + 1.00 * gy
            )

            candidates.append(
                (
                    dist,
                    j,
                )
            )

        candidates.sort(
            key=lambda x: x[0]
        )

        for _, j in candidates[
            :top_k
        ]:

            if i < j:
                pairs.add(
                    (i, j)
                )
            else:
                pairs.add(
                    (j, i)
                )

    # -------------------------------------------------------------------------
    # 4) Pixel-aware neighbors.
    #
    # This does not replace geometry. It adds candidates that look visually
    # connected under the same blur/ink representation used by the relation
    # head, which helps recover positives missed by purely geometric KNN.
    # -------------------------------------------------------------------------

    if pixel_context is not None:

        med_h_px = median(
            [
                w.box.h
                for w in words
                if w.box.h > 0
            ],
            10.0,
        )

        pixel_cache = [
            word_pixel_features(
                w,
                pixel_context,
                med_h_px,
            )
            for w in words
        ]

        # The V3 geometric pass is already O(N^2). Reuse it as the pixel
        # search pool instead of evaluating pixel features for every possible
        # pair again. A wider pool improves recall without multiplying the
        # quadratic work by another full pass.
        pixel_pool_k = max(
            int(pixel_top_k),
            int(top_k) * 4,
        )

        for i, a in enumerate(words):

            geom_pool = []

            for j, b in enumerate(words):

                if i == j:
                    continue

                dx = abs(a.box.cx - b.box.cx) / max(page_box.w, 1.0)
                dy = abs(a.box.cy - b.box.cy) / max(page_box.h, 1.0)
                gx = gap_x(a.box, b.box) / max(page_box.w, 1.0)
                gy = gap_y(a.box, b.box) / max(page_box.h, 1.0)

                geom = (
                    1.00 * dx
                    + 1.20 * dy
                    + 0.70 * gx
                    + 0.90 * gy
                )

                geom_pool.append((geom, j))

            geom_pool.sort(key=lambda x: x[0])

            scored = []

            for geom, j in geom_pool[:pixel_pool_k]:

                ps = quick_pixel_link_score(
                    i,
                    j,
                    words,
                    pixel_context,
                    pixel_cache,
                    med_h_px,
                )

                score = geom - 0.40 * ps
                scored.append((score, -ps, j))

            scored.sort(key=lambda x: (x[0], x[1], x[2]))

            for _, _, j in scored[:max(1, int(pixel_top_k))]:

                if i < j:
                    pairs.add((i, j))
                else:
                    pairs.add((j, i))

    # -------------------------------------------------------------------------
    # 5) Nearby vertical words with x support
    # -------------------------------------------------------------------------

    for i in range(n):

        a = words[i]

        for j in range(
            i + 1,
            n,
        ):

            b = words[j]

            gy = gap_y(
                a.box,
                b.box,
            )

            if gy > 10.0 * med_h:
                continue

            xo = x_overlap(
                a.box,
                b.box,
            )

            x_center_gap = abs(
                a.box.cx -
                b.box.cx
            )

            if (
                xo >= 0.10
                or x_center_gap <= 2.0 * max(
                    a.box.w,
                    b.box.w,
                    1.0,
                )
            ):

                pairs.add(
                    (i, j)
                )

    return sorted(
        pairs
    )


# =============================================================================
# Symmetric relation model
# =============================================================================

class SymmetricRelationHeadFactory:

    @staticmethod
    def build(config):

        import torch.nn as nn

        projection = nn.Sequential(
            nn.Linear(
                config.hidden_size,
                config.projection_size,
            ),
            nn.GELU(),
            nn.LayerNorm(
                config.projection_size
            ),
        )

        pair_input = (
            3 * config.projection_size
            + 1
            + config.geometry_dim
            + config.pixel_dim
        )

        classifier = nn.Sequential(
            nn.Linear(
                pair_input,
                config.width,
            ),
            nn.GELU(),
            nn.LayerNorm(
                config.width
            ),
            nn.Dropout(
                config.dropout
            ),

            nn.Linear(
                config.width,
                config.bottleneck,
            ),
            nn.GELU(),

            nn.Dropout(
                config.dropout
            ),

            nn.Linear(
                config.bottleneck,
                1,
            ),
        )

        class Model(nn.Module):

            def __init__(self):

                super(Model, self).__init__()

                self.projection = projection
                self.classifier = classifier

            def forward(
                self,
                ha,
                hb,
                geometry,
                pixel_features,
            ):

                za = self.projection(
                    ha
                )

                zb = self.projection(
                    hb
                )

                sum_z = (
                    za + zb
                )

                abs_diff = torch.abs(
                    za - zb
                )

                product = (
                    za * zb
                )

                cosine = torch.nn.functional.cosine_similarity(
                    za,
                    zb,
                    dim=-1,
                    eps=1e-8,
                ).unsqueeze(
                    -1
                )

                x = torch.cat(
                    [
                        sum_z,
                        abs_diff,
                        product,
                        cosine,
                        geometry,
                        pixel_features,
                    ],
                    dim=-1,
                )

                return self.classifier(
                    x
                )

        import torch

        # Make torch visible inside Model.forward.
        _ = torch

        return Model()


# =============================================================================
# Candidate pair tensors
# =============================================================================

def make_pair_geometry(
    words,
    pairs,
    page_box,
):

    import torch

    med_h = median(
        [
            w.box.h
            for w in words
            if w.box.h > 0
        ],
        10.0,
    )

    geometry = []

    for i, j in pairs:

        geometry.append(
            geometry_features(
                words[i],
                words[j],
                page_box,
                med_h,
            )
        )

    if not geometry:

        return torch.zeros(
            (
                0,
                GEOMETRY_DIM,
            ),
            dtype=torch.float32,
        )

    return torch.tensor(
        geometry,
        dtype=torch.float32,
    )


def pair_labels(
    words,
    pairs,
):

    import torch

    labels = []

    for i, j in pairs:

        a = words[i].segment_id
        b = words[j].segment_id

        labels.append(
            1.0
            if (
                a is not None
                and b is not None
                and a == b
            )
            else 0.0
        )

    return torch.tensor(
        labels,
        dtype=torch.float32,
    )


# =============================================================================
# Gold positive pair generation
# =============================================================================

def all_gold_positive_pairs(words):

    by_segment = defaultdict(list)

    for i, w in enumerate(words):

        if w.segment_id is not None:

            by_segment[
                w.segment_id
            ].append(i)

    pairs = []

    for sid, ids in by_segment.items():

        if len(ids) < 2:
            continue

        ids = sorted(ids)

        for x in range(
            len(ids)
        ):

            for y in range(
                x + 1,
                len(ids),
            ):

                pairs.append(
                    (
                        ids[x],
                        ids[y],
                    )
                )

    return pairs


# =============================================================================
# Training pair sampling
# =============================================================================

def balanced_indices(
    labels,
    seed,
    negative_ratio=3,
    max_positive=4000,
):

    rng = random.Random(
        seed
    )

    positive = [
        i
        for i, y in enumerate(labels)
        if y == 1
    ]

    negative = [
        i
        for i, y in enumerate(labels)
        if y == 0
    ]

    if len(positive) > max_positive:

        rng.shuffle(
            positive
        )

        positive = positive[
            :max_positive
        ]

    target_negative = min(
        len(negative),
        negative_ratio * len(positive),
    )

    rng.shuffle(
        negative
    )

    negative = negative[
        :target_negative
    ]

    selected = (
        positive
        + negative
    )

    rng.shuffle(
        selected
    )

    return selected


# =============================================================================
# Page preparation
# =============================================================================

def prepare_page(
    tokenizer,
    model,
    image_path,
    annotation_path,
    device,
    max_length,
    use_gold_candidate_injection=False,
    top_k=DEFAULT_TOP_K,
    pixel_top_k=24,
):

    image = Image.open(
        str(image_path)
    ).convert("RGB")

    words, entities = load_page_annotation(
        annotation_path
    )

    if not words:
        return None

    page_box = union_boxes(
        w.box
        for w in words
    )

    pixel_context = build_pixel_context(
        image,
        page_box,
    )

    covered = extract_word_hidden(
        tokenizer,
        model,
        image,
        words,
        page_box,
        device,
        max_length,
    )

    # Base candidate graph.
    pairs = candidate_pairs(
        words,
        page_box,
        top_k=top_k,
        pixel_context=pixel_context,
        pixel_top_k=pixel_top_k,
    )

    # During training we explicitly add gold positive pairs.
    #
    # This removes an unnecessary training bottleneck:
    # the model should actually see true positive relations,
    # rather than relying entirely on geometry heuristics.
    if use_gold_candidate_injection:

        gold_positive = all_gold_positive_pairs(
            words
        )

        pair_set = set(
            pairs
        )

        for pair in gold_positive:

            if pair not in pair_set:

                pairs.append(
                    pair
                )

        pairs.sort()

    labels = pair_labels(
        words,
        pairs,
    )

    geometry = make_pair_geometry(
        words,
        pairs,
        page_box,
    )

    pixel_features = make_pair_pixel_features(
        words,
        pairs,
        pixel_context,
    )

    hidden_dim = (
        words[0].hidden.shape[-1]
    )

    hidden = torch_stack_hidden(
        words,
        hidden_dim,
    )

    try:
        image.close()
    except Exception:
        pass

    return {
        "words": words,
        "entities": entities,
        "page_box": page_box,
        "pairs": pairs,
        "labels": labels,
        "geometry": geometry,
        "pixel_features": pixel_features,
        "hidden": hidden,
        "covered_words": covered,
    }


def torch_stack_hidden(
    words,
    hidden_dim,
):

    import torch

    values = []

    for w in words:

        if w.hidden is None:

            values.append(
                torch.zeros(
                    hidden_dim,
                    dtype=torch.float32,
                )
            )

        else:

            values.append(
                w.hidden.float()
            )

    return torch.stack(
        values,
        dim=0,
    )


# =============================================================================
# Relation forward helper
# =============================================================================

def relation_batch(
    head,
    hidden,
    pair_indices,
    geometry,
    pixel_features,
    device,
):

    import torch

    if pair_indices.numel() == 0:

        return torch.zeros(
            (
                0,
            ),
            dtype=torch.float32,
            device=device,
        )

    idx_a = pair_indices[
        :,
        0,
    ]

    idx_b = pair_indices[
        :,
        1,
    ]

    ha = hidden[
        idx_a
    ].to(device)

    hb = hidden[
        idx_b
    ].to(device)

    g = geometry.to(
        device
    )

    pf = pixel_features.to(
        device
    )

    return head(
        ha,
        hb,
        g,
        pf,
    ).squeeze(
        -1
    )


def predict_pairs(
    head,
    hidden,
    pairs,
    geometry,
    pixel_features,
    device,
    batch_size=2048,
):

    import torch

    if not pairs:

        return []

    pair_tensor = torch.tensor(
        pairs,
        dtype=torch.long,
    )

    out = []

    head.eval()

    with torch.no_grad():

        for start in range(
            0,
            len(pairs),
            batch_size,
        ):

            end = min(
                len(pairs),
                start + batch_size,
            )

            pair_batch = pair_tensor[
                start:end
            ]

            geom_batch = geometry[
                start:end
            ]

            logits = relation_batch(
                head,
                hidden,
                pair_batch,
                geom_batch,
                pixel_features[start:end],
                device,
            )

            probs = torch.sigmoid(
                logits
            ).cpu().tolist()

            out.extend(
                float(x)
                for x in probs
            )

    return out


# =============================================================================
# Metrics
# =============================================================================

def binary_metrics(
    y_true,
    y_score,
    threshold,
):

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for y, p in zip(
        y_true,
        y_score,
    ):

        pred = (
            1
            if p >= threshold
            else 0
        )

        if y == 1 and pred == 1:
            tp += 1

        elif y == 0 and pred == 1:
            fp += 1

        elif y == 1 and pred == 0:
            fn += 1

        else:
            tn += 1

    precision = safe_div(
        tp,
        tp + fp,
    )

    recall = safe_div(
        tp,
        tp + fn,
    )

    f1 = safe_div(
        2.0 * precision * recall,
        precision + recall,
    )

    accuracy = safe_div(
        tp + tn,
        tp + tn + fp + fn,
    )

    return {
        "threshold": float(
            threshold
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def roc_auc_score_manual(
    y_true,
    y_score,
):

    if not y_true:
        return 0.0

    pos = sum(
        1
        for y in y_true
        if y == 1
    )

    neg = sum(
        1
        for y in y_true
        if y == 0
    )

    if pos == 0 or neg == 0:
        return 0.0

    pairs = sorted(
        zip(
            y_score,
            y_true,
        ),
        key=lambda x: x[0],
        reverse=True,
    )

    tp = 0.0
    fp = 0.0

    prev_tpr = 0.0
    prev_fpr = 0.0

    auc = 0.0

    idx = 0

    while idx < len(pairs):

        score = pairs[
            idx
        ][0]

        group_pos = 0
        group_neg = 0

        while (
            idx < len(pairs)
            and pairs[idx][0] == score
        ):

            if pairs[idx][1] == 1:
                group_pos += 1
            else:
                group_neg += 1

            idx += 1

        tp += group_pos
        fp += group_neg

        tpr = tp / float(pos)
        fpr = fp / float(neg)

        auc += (
            fpr - prev_fpr
        ) * (
            tpr + prev_tpr
        ) * 0.5

        prev_tpr = tpr
        prev_fpr = fpr

    return float(
        auc
    )


def average_precision_score_manual(
    y_true,
    y_score,
):

    if not y_true:
        return 0.0

    total_pos = sum(
        1
        for y in y_true
        if y == 1
    )

    if total_pos == 0:
        return 0.0

    pairs = sorted(
        zip(
            y_score,
            y_true,
        ),
        key=lambda x: x[0],
        reverse=True,
    )

    tp = 0
    ap = 0.0

    for idx, (_, y) in enumerate(
        pairs,
        start=1,
    ):

        if y == 1:

            tp += 1

            precision = (
                tp /
                float(idx)
            )

            ap += precision

    return float(
        ap /
        float(total_pos)
    )


# =============================================================================
# Candidate recall
# =============================================================================

def candidate_recall(
    words,
    pairs,
):

    gold_positive = set(
        all_gold_positive_pairs(
            words
        )
    )

    candidate_set = set(
        pairs
    )

    covered = sum(
        1
        for p in gold_positive
        if p in candidate_set
    )

    return {
        "gold_positive_pairs": len(
            gold_positive
        ),
        "covered_positive_pairs": covered,
        "recall": safe_div(
            covered,
            len(gold_positive),
        ),
    }


# =============================================================================
# Raw MLP diagnostic
# =============================================================================

def raw_mlp_diagnosis(
    head,
    pages,
    device,
):

    all_labels = []
    all_probs = []

    page_stats = []

    for page in pages:

        probs = predict_pairs(
            head,
            page["hidden"],
            page["pairs"],
            page["geometry"],
            page["pixel_features"],
            device,
        )

        labels = page[
            "labels"
        ].tolist()

        if not labels:
            continue

        all_labels.extend(
            labels
        )

        all_probs.extend(
            probs
        )

        page_stats.append(
            {
                "stem": page["stem"],
                "labels": labels,
                "probs": probs,
            }
        )

    print()
    print("=" * 60)
    print("RAW WORD-LEVEL MLP DIAGNOSIS")
    print("=" * 60)

    if not all_labels:

        print(
            "No usable pairs."
        )

        return None

    pos_probs = [
        p
        for y, p in zip(
            all_labels,
            all_probs,
        )
        if y == 1
    ]

    neg_probs = [
        p
        for y, p in zip(
            all_labels,
            all_probs,
        )
        if y == 0
    ]

    pos = sum(
        1
        for y in all_labels
        if y == 1
    )

    neg = sum(
        1
        for y in all_labels
        if y == 0
    )

    auc = roc_auc_score_manual(
        all_labels,
        all_probs,
    )

    ap = average_precision_score_manual(
        all_labels,
        all_probs,
    )

    print(
        "Usable candidate pairs          : {}".format(
            len(all_labels)
        )
    )

    print(
        "Positive pairs                  : {}".format(
            pos
        )
    )

    print(
        "Negative pairs                  : {}".format(
            neg
        )
    )

    print(
        "Positive ratio                  : {:.4f}".format(
            safe_div(
                pos,
                len(all_labels),
            )
        )
    )

    print()
    print(
        "Positive probability:"
    )

    print(
        "  mean                          : {:.4f}".format(
            sum(pos_probs) /
            max(1, len(pos_probs))
        )
    )

    print(
        "  median                        : {:.4f}".format(
            median(
                pos_probs,
                0.0,
            )
        )
    )

    print(
        "  p10                           : {:.4f}".format(
            percentile(
                pos_probs,
                0.10,
            )
        )
    )

    print(
        "  p25                           : {:.4f}".format(
            percentile(
                pos_probs,
                0.25,
            )
        )
    )

    print(
        "  p75                           : {:.4f}".format(
            percentile(
                pos_probs,
                0.75,
            )
        )
    )

    print(
        "  p90                           : {:.4f}".format(
            percentile(
                pos_probs,
                0.90,
            )
        )
    )

    print()
    print(
        "Negative probability:"
    )

    print(
        "  mean                          : {:.4f}".format(
            sum(neg_probs) /
            max(1, len(neg_probs))
        )
    )

    print(
        "  median                        : {:.4f}".format(
            median(
                neg_probs,
                0.0,
            )
        )
    )

    print(
        "  p10                           : {:.4f}".format(
            percentile(
                neg_probs,
                0.10,
            )
        )
    )

    print(
        "  p25                           : {:.4f}".format(
            percentile(
                neg_probs,
                0.25,
            )
        )
    )

    print(
        "  p75                           : {:.4f}".format(
            percentile(
                neg_probs,
                0.75,
            )
        )
    )

    print(
        "  p90                           : {:.4f}".format(
            percentile(
                neg_probs,
                0.90,
            )
        )
    )

    print()
    print(
        "ROC-AUC                         : {:.4f}".format(
            auc
        )
    )

    print(
        "Average Precision               : {:.4f}".format(
            ap
        )
    )

    print()
    print(
        "RAW MLP THRESHOLD SWEEP"
    )

    print(
        "threshold  precision     recall         F1"
    )

    print(
        "-" * 50
    )

    thresholds = [
        0.20,
        0.30,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    sweep = []

    for t in thresholds:

        m = binary_metrics(
            all_labels,
            all_probs,
            t,
        )

        sweep.append(
            m
        )

        print(
            "{:9.2f} {:10.4f} {:10.4f} {:10.4f}".format(
                t,
                m["precision"],
                m["recall"],
                m["f1"],
            )
        )

    best = max(
        sweep,
        key=lambda x:
        x["f1"]
    )

    print()
    print(
        "Best pooled raw F1              : {:.4f} @ {:.2f}".format(
            best["f1"],
            best["threshold"],
        )
    )

    return {
        "num_pairs": len(
            all_labels
        ),
        "positive": pos,
        "negative": neg,
        "roc_auc": auc,
        "average_precision": ap,
        "threshold_sweep": sweep,
        "best": best,
        "pages": page_stats,
    }


# =============================================================================
# Clustering
#
# This is intentionally conservative.
#
# It uses:
#   - base probability threshold
#   - mutual top-k
#   - local confidence relative to each node's strongest edges
#
# It operates on WORD nodes directly.
# =============================================================================

def can_link_words(
    a,
    b,
    page_box,
    med_h,
    pixel_score=0.0,
    pixel_gate=0.12,
):

    gy = gap_y(
        a.box,
        b.box,
    )

    gx = gap_x(
        a.box,
        b.box,
    )

    xo = x_overlap(
        a.box,
        b.box,
    )

    yo = y_overlap(
        a.box,
        b.box,
    )

    # Same line: fairly permissive.
    if (
        a.row_id == b.row_id
        and gx <= 8.0 * med_h
    ):
        return True

    # Nearby lines with horizontal alignment.
    if (
        gy <= 10.0 * med_h
        and (
            xo >= 0.10
            or abs(
                a.box.cx -
                b.box.cx
            )
            <= 2.5 * max(
                a.box.w,
                b.box.w,
                1.0,
            )
        )
    ):
        return True

    # Nearby vertically overlapping boxes.
    if (
        gy <= 6.0 * med_h
        and yo >= 0.10
    ):
        return True

    # A word can connect across a moderately larger gap if x aligned.
    if (
        gy <= 14.0 * med_h
        and abs(
            a.box.cx -
            b.box.cx
        )
        <= 1.5 * max(
            a.box.w,
            b.box.w,
            1.0,
        )
    ):
        return True

    # Pixel evidence may rescue a pair that is a little too far apart for the
    # hand-written geometry prior, but only when the blurred image shows a
    # genuine bridge and the separation is still locally plausible.
    if (
        pixel_score >= pixel_gate
        and gx <= 12.0 * med_h
        and gy <= 18.0 * med_h
    ):
        return True

    return False


def adaptive_edges(
    words,
    pairs,
    probs,
    pixel_features=None,
    threshold=0.50,
    top_k=8,
    pixel_order_weight=0.10,
):

    # Top probabilities incident to every word.
    incident = defaultdict(list)

    edge_records = []

    if pixel_features is None:
        pixel_features = [None] * len(pairs)

    for (i, j), p, pf in zip(
        pairs,
        probs,
        pixel_features,
    ):

        ps = pixel_link_score(pf)

        incident[i].append(
            (
                p,
                j,
            )
        )

        incident[j].append(
            (
                p,
                i,
            )
        )

        edge_records.append(
            (
                p,
                i,
                j,
                ps,
            )
        )

    top_by_node = {}

    for i, values in incident.items():

        values = sorted(
            values,
            key=lambda x:
            x[0],
            reverse=True,
        )

        top_by_node[i] = values[
            :top_k
        ]

    accepted = []

    for p, i, j, ps in sorted(
        edge_records,
        key=lambda x: (
            x[0] + pixel_order_weight * x[3],
            x[0],
        ),
        reverse=True,
    ):

        if p < threshold:
            continue

        nei_i = top_by_node.get(
            i,
            [],
        )

        nei_j = top_by_node.get(
            j,
            [],
        )

        rank_i = None
        rank_j = None
        best_i = 0.0
        best_j = 0.0

        for rank, (pp, node) in enumerate(
            nei_i
        ):

            if node == j:
                rank_i = rank
                best_i = pp
                break

        for rank, (pp, node) in enumerate(
            nei_j
        ):

            if node == i:
                rank_j = rank
                best_j = pp
                break

        if rank_i is None or rank_j is None:
            continue

        # Mutual local support.
        #
        # We do NOT demand that the edge is #1 for both endpoints.
        # A node often legitimately belongs to a multi-word entity.
        if rank_i >= top_k or rank_j >= top_k:
            continue

        # Prevent a weak edge from attaching to an endpoint whose
        # neighborhood is overwhelmingly stronger.
        local_ratio_i = safe_div(
            p,
            best_i,
        )

        local_ratio_j = safe_div(
            p,
            best_j,
        )

        if (
            local_ratio_i < 0.45
            and local_ratio_j < 0.45
        ):
            continue

        accepted.append(
            (
                p,
                i,
                j,
                ps,
            )
        )

    return accepted


def cluster_words(
    words,
    pairs,
    probs,
    pixel_features,
    page_box,
    threshold=0.50,
    top_k=8,
    pixel_gate=0.12,
    pixel_order_weight=0.10,
):

    n = len(words)

    if n == 0:
        return []

    for w in words:
        if w.row_id < 0:
            pass

    med_h = median(
        [
            w.box.h
            for w in words
            if w.box.h > 0
        ],
        10.0,
    )

    accepted = adaptive_edges(
        words,
        pairs,
        probs,
        pixel_features=pixel_features,
        threshold=threshold,
        top_k=top_k,
        pixel_order_weight=pixel_order_weight,
    )

    # -------------------------------------------------------------------------
    # Union-find.
    # -------------------------------------------------------------------------

    parent = list(
        range(n)
    )

    size = [
        1
        for _ in range(n)
    ]

    def find(x):

        while parent[x] != x:

            parent[x] = parent[
                parent[x]
            ]

            x = parent[x]

        return x

    def union(a, b):

        ra = find(a)
        rb = find(b)

        if ra == rb:
            return

        if size[ra] < size[rb]:
            ra, rb = rb, ra

        parent[rb] = ra

        size[ra] += size[rb]

    # -------------------------------------------------------------------------
    # Sort high-confidence edges first.
    # -------------------------------------------------------------------------

    for p, i, j, ps in accepted:

        if not can_link_words(
            words[i],
            words[j],
            page_box,
            med_h,
            pixel_score=ps,
            pixel_gate=pixel_gate,
        ):
            continue

        ri = find(i)
        rj = find(j)

        if ri == rj:
            continue

        # Very large components require stronger edges.
        component_size = (
            size[ri] +
            size[rj]
        )

        required = threshold

        if component_size >= 8:
            required = max(
                required,
                0.55,
            )

        if component_size >= 20:
            required = max(
                required,
                0.60,
            )

        if component_size >= 40:
            required = max(
                required,
                0.65,
            )

        if p < required:
            continue

        union(
            ri,
            rj,
        )

    groups = defaultdict(list)

    for i in range(n):

        groups[
            find(i)
        ].append(i)

    output = list(
        groups.values()
    )

    output.sort(
        key=lambda ids:
        min(
            (
                words[i].box.y0,
                words[i].box.x0,
            )
            for i in ids
        )
    )

    return output


# =============================================================================
# Segment output
# =============================================================================

def groups_to_segments(
    groups,
    words,
):

    segments = []

    for sid, ids in enumerate(
        groups
    ):

        ids = sorted(
            ids,
            key=lambda i: (
                words[i].box.y0,
                words[i].box.x0,
            )
        )

        box = union_boxes(
            words[i].box
            for i in ids
        )

        segments.append(
            {
                "segment_id": sid,
                "word_ids": ids,
                "text": " ".join(
                    words[i].text
                    for i in ids
                ),
                "bbox": [
                    box.x0,
                    box.y0,
                    box.x1,
                    box.y1,
                ],
            }
        )

    return segments


# =============================================================================
# Final word-level pairwise evaluation
# =============================================================================

def pairwise_segment_metrics(
    words,
    segments,
):

    n = len(words)

    pred_label = [
        -1
        for _ in range(n)
    ]

    for seg in segments:

        sid = int(
            seg["segment_id"]
        )

        for wid in seg["word_ids"]:

            if 0 <= wid < n:

                pred_label[
                    wid
                ] = sid

    valid = [
        i
        for i in range(n)
        if (
            words[i].segment_id is not None
            and pred_label[i] >= 0
        )
    ]

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for aa in range(
        len(valid)
    ):

        i = valid[aa]

        for bb in range(
            aa + 1,
            len(valid),
        ):

            j = valid[bb]

            gold_same = (
                words[i].segment_id
                ==
                words[j].segment_id
            )

            pred_same = (
                pred_label[i]
                ==
                pred_label[j]
            )

            if gold_same and pred_same:
                tp += 1

            elif not gold_same and pred_same:
                fp += 1

            elif gold_same and not pred_same:
                fn += 1

            else:
                tn += 1

    precision = safe_div(
        tp,
        tp + fp,
    )

    recall = safe_div(
        tp,
        tp + fn,
    )

    f1 = safe_div(
        2.0 * precision * recall,
        precision + recall,
    )

    return {
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "coverage": safe_div(
            len(valid),
            n,
        ),
        "num_pred_segments": len(
            segments
        ),
    }


# =============================================================================
# Visualization
# =============================================================================

def draw_pred(
    ax,
    image,
    segments,
    title,
):

    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title)

    for sid, seg in enumerate(
        segments
    ):

        x0, y0, x1, y1 = (
            seg["bbox"]
        )

        rect = Rectangle(
            (
                x0,
                y0,
            ),
            x1 - x0,
            y1 - y0,
            fill=False,
            linewidth=2.0,
        )

        ax.add_patch(
            rect
        )

        ax.text(
            x0 + 2,
            y0 + 2,
            "S{}".format(
                sid
            ),
            fontsize=8,
            bbox=dict(
                facecolor="white",
                alpha=0.75,
                pad=1,
            ),
        )


def draw_gold(
    ax,
    image,
    words,
    title,
):

    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title)

    groups = defaultdict(list)

    for w in words:

        if w.segment_id is not None:

            groups[
                str(w.segment_id)
            ].append(
                w.idx
            )

    keys = list(
        groups.keys()
    )

    def key_sorter(x):

        try:
            return (
                0,
                int(x),
            )
        except Exception:
            return (
                1,
                str(x),
            )

    keys.sort(
        key=key_sorter
    )

    for gid, sid in enumerate(
        keys
    ):

        ids = groups[
            sid
        ]

        box = union_boxes(
            words[i].box
            for i in ids
        )

        rect = Rectangle(
            (
                box.x0,
                box.y0,
            ),
            box.w,
            box.h,
            fill=False,
            linewidth=2.0,
        )

        ax.add_patch(
            rect
        )

        ax.text(
            box.x0 + 2,
            box.y0 + 2,
            "G{}".format(
                gid
            ),
            fontsize=8,
            bbox=dict(
                facecolor="white",
                alpha=0.75,
                pad=1,
            ),
        )


def visualize_page(
    image,
    words,
    segments,
    out_path,
    title,
    metrics=None,
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            15,
            9,
        ),
    )

    metric_text = ""

    if metrics:

        metric_text = (
            " | F1={:.3f} P={:.3f} R={:.3f}".format(
                metrics[
                    "pairwise_f1"
                ],
                metrics[
                    "pairwise_precision"
                ],
                metrics[
                    "pairwise_recall"
                ],
            )
        )

    draw_pred(
        axes[0],
        image,
        segments,
        "Predicted segments" +
        metric_text,
    )

    draw_gold(
        axes[1],
        image,
        words,
        "Gold FUNSD form.id",
    )

    fig.suptitle(
        title
    )

    fig.tight_layout()

    out_path = Path(
        out_path
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        str(out_path),
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# Files
# =============================================================================

def list_stems(
    image_dir,
    annotation_dir,
):

    image_dir = Path(
        image_dir
    )

    annotation_dir = Path(
        annotation_dir
    )

    stems = []

    for image_path in sorted(
        image_dir.glob(
            "*.png"
        )
    ):

        annotation_path = (
            annotation_dir /
            (
                image_path.stem +
                ".json"
            )
        )

        if annotation_path.exists():

            stems.append(
                image_path.stem
            )

    return stems


# =============================================================================
# Head IO
# =============================================================================

def save_head(
    path,
    head,
    config,
):

    import torch

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "state_dict":
                head.state_dict(),

            "hidden_size":
                config.hidden_size,

            "projection_size":
                config.projection_size,

            "geometry_dim":
                config.geometry_dim,

            "pixel_dim":
                config.pixel_dim,

            "width":
                config.width,

            "bottleneck":
                config.bottleneck,

            "dropout":
                config.dropout,
        },
        str(path),
    )

    print(
        "[SAVE] {}".format(
            path
        )
    )


def load_head(
    path,
    device,
):

    import torch

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    ckpt = torch.load(
        str(path),
        map_location=device,
    )

    config = RelationHeadConfig(
        hidden_size=int(
            ckpt[
                "hidden_size"
            ]
        ),
        projection_size=int(
            ckpt.get(
                "projection_size",
                HIDDEN_PROJECTION,
            )
        ),
        geometry_dim=int(
            ckpt.get(
                "geometry_dim",
                GEOMETRY_DIM,
            )
        ),
        pixel_dim=int(
            ckpt.get(
                "pixel_dim",
                PIXEL_DIM,
            )
        ),
        width=int(
            ckpt.get(
                "width",
                256,
            )
        ),
        bottleneck=int(
            ckpt.get(
                "bottleneck",
                64,
            )
        ),
        dropout=float(
            ckpt.get(
                "dropout",
                0.10,
            )
        ),
    )

    head = (
        SymmetricRelationHeadFactory
        .build(
            config
        )
        .to(device)
    )

    head.load_state_dict(
        ckpt[
            "state_dict"
        ]
    )

    head.eval()

    print(
        "[HEAD] loaded {}".format(
            path
        )
    )

    print(
        "[HEAD] word-level symmetric relation head"
    )

    return head, config


# =============================================================================
# Training
# =============================================================================

def build_train_validation_split(
    stems,
    val_ratio,
    seed,
):

    stems = list(
        stems
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        stems
    )

    if len(stems) <= 1:
        return stems, []

    val_count = max(
        1,
        int(
            round(
                len(stems) *
                val_ratio
            )
        ),
    )

    val = stems[
        :val_count
    ]

    train = stems[
        val_count:
    ]

    return train, val


def evaluate_pages_raw(
    head,
    pages,
    device,
):

    all_labels = []
    all_probs = []

    for page in pages:

        probs = predict_pairs(
            head,
            page["hidden"],
            page["pairs"],
            page["geometry"],
            page["pixel_features"],
            device,
        )

        labels = page[
            "labels"
        ].tolist()

        all_labels.extend(
            labels
        )

        all_probs.extend(
            probs
        )

    if not all_labels:

        return {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "auc": 0.0,
            "ap": 0.0,
        }

    auc = roc_auc_score_manual(
        all_labels,
        all_probs,
    )

    ap = average_precision_score_manual(
        all_labels,
        all_probs,
    )

    sweep = []

    for threshold in [
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
    ]:

        sweep.append(
            binary_metrics(
                all_labels,
                all_probs,
                threshold,
            )
        )

    best = max(
        sweep,
        key=lambda x:
        x["f1"],
    )

    return {
        "f1": best["f1"],
        "precision": best[
            "precision"
        ],
        "recall": best[
            "recall"
        ],
        "threshold": best[
            "threshold"
        ],
        "auc": auc,
        "ap": ap,
    }


def train(args):

    import torch
    import torch.nn as nn

    device = args.device

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model_path = Path(
        args.model_path
    )

    tokenizer, model = load_layoutlmv3(
        model_path,
        device,
    )

    train_images = Path(
        args.train_images
    )

    train_annotations = Path(
        args.train_annotations
    )

    stems = list_stems(
        train_images,
        train_annotations,
    )

    if not stems:

        raise RuntimeError(
            "No training pages found."
        )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # If no separate validation set is provided, make a validation split
    # INSIDE the training set.
    #
    # This avoids using the 50-page test set for model selection.
    # -------------------------------------------------------------------------

    if args.val_images and args.val_annotations:

        val_images = Path(
            args.val_images
        )

        val_annotations = Path(
            args.val_annotations
        )

        train_stems = list(
            stems
        )

        val_stems = list_stems(
            val_images,
            val_annotations,
        )

    else:

        train_stems, val_stems = (
            build_train_validation_split(
                stems,
                args.val_ratio,
                args.seed,
            )
        )

        val_images = train_images
        val_annotations = train_annotations

    print()
    print(
        "[TRAIN] train pages={}".format(
            len(train_stems)
        )
    )

    print(
        "[TRAIN] val pages={}".format(
            len(val_stems)
        )
    )

    # -------------------------------------------------------------------------
    # Build head.
    # -------------------------------------------------------------------------

    hidden_size = int(
        model.config.hidden_size
    )

    config = RelationHeadConfig(
        hidden_size=hidden_size,
        projection_size=args.projection_size,
        geometry_dim=GEOMETRY_DIM,
        pixel_dim=PIXEL_DIM,
        width=args.head_width,
        bottleneck=args.head_bottleneck,
        dropout=args.dropout,
    )

    head = (
        SymmetricRelationHeadFactory
        .build(
            config
        )
        .to(device)
    )

    trainable_params = sum(
        p.numel()
        for p in head.parameters()
        if p.requires_grad
    )

    backbone_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "[TRAIN] frozen backbone parameters: {:,}".format(
            backbone_params
        )
    )

    print(
        "[TRAIN] trainable relation-head parameters: {:,}".format(
            trainable_params
        )
    )

    # -------------------------------------------------------------------------
    # Precompute word features ONCE.
    #
    # Backbone is frozen, therefore there is no reason to run LayoutLMv3 again
    # every epoch.
    # -------------------------------------------------------------------------

    print()
    print(
        "[CACHE] extracting frozen LayoutLMv3 word features..."
    )

    train_pages = []

    for idx, stem in enumerate(
        train_stems,
        start=1,
    ):

        print(
            "[CACHE TRAIN {:03d}/{}] {}".format(
                idx,
                len(train_stems),
                stem,
            )
        )

        page = prepare_page(
            tokenizer,
            model,
            train_images / (
                stem + ".png"
            ),
            train_annotations / (
                stem + ".json"
            ),
            device,
            args.max_length,
            use_gold_candidate_injection=True,
            top_k=args.top_k,
            pixel_top_k=args.pixel_top_k,
        )

        if page is None:
            continue

        page["stem"] = stem

        train_pages.append(
            page
        )

    val_pages = []

    for idx, stem in enumerate(
        val_stems,
        start=1,
    ):

        print(
            "[CACHE VAL {:03d}/{}] {}".format(
                idx,
                len(val_stems),
                stem,
            )
        )

        page = prepare_page(
            tokenizer,
            model,
            val_images / (
                stem + ".png"
            ),
            val_annotations / (
                stem + ".json"
            ),
            device,
            args.max_length,
            use_gold_candidate_injection=False,
            top_k=args.top_k,
            pixel_top_k=args.pixel_top_k,
        )

        if page is None:
            continue

        page["stem"] = stem

        val_pages.append(
            page
        )

    # -------------------------------------------------------------------------
    # Cache statistics.
    # -------------------------------------------------------------------------

    print()
    print(
        "[CACHE] TRAIN"
    )

    total_train_pairs = sum(
        len(
            p["pairs"]
        )
        for p in train_pages
    )

    total_train_pos = sum(
        int(
            p["labels"].sum().item()
        )
        for p in train_pages
    )

    total_train_gold_pos = sum(
        len(
            all_gold_positive_pairs(
                p["words"]
            )
        )
        for p in train_pages
    )

    print(
        "  pages                       : {}".format(
            len(train_pages)
        )
    )

    print(
        "  candidate pairs             : {}".format(
            total_train_pairs
        )
    )

    print(
        "  candidate positives         : {}".format(
            total_train_pos
        )
    )

    print(
        "  gold positive pairs         : {}".format(
            total_train_gold_pos
        )
    )

    print(
        "  positive coverage           : {:.4f}".format(
            safe_div(
                total_train_pos,
                total_train_gold_pos,
            )
        )
    )

    # -------------------------------------------------------------------------
    # Optimizer.
    # -------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    best_ap = -1.0

    output_path = Path(
        args.head_out
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Train epochs.
    # -------------------------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        head.train()

        epoch_loss = 0.0
        epoch_steps = 0
        epoch_pairs = 0
        epoch_pos = 0

        order = list(
            range(
                len(
                    train_pages
                )
            )
        )

        random.Random(
            args.seed + epoch
        ).shuffle(
            order
        )

        for page_index in order:

            page = train_pages[
                page_index
            ]

            labels = page[
                "labels"
            ]

            selected = balanced_indices(
                labels.tolist(),
                seed=(
                    args.seed
                    + 1000 * epoch
                    + page_index
                ),
                negative_ratio=args.negative_ratio,
                max_positive=args.max_positive_per_page,
            )

            if not selected:
                continue

            rng = random.Random(
                args.seed
                + epoch * 17
                + page_index
            )

            rng.shuffle(
                selected
            )

            pairs = page[
                "pairs"
            ]

            geometry = page[
                "geometry"
            ]

            hidden = page[
                "hidden"
            ]

            for start in range(
                0,
                len(selected),
                args.batch_size,
            ):

                chunk = selected[
                    start:
                    start +
                    args.batch_size
                ]

                pair_indices = torch.tensor(
                    [
                        pairs[i]
                        for i in chunk
                    ],
                    dtype=torch.long,
                )

                geom = geometry[
                    chunk
                ]

                y = labels[
                    chunk
                ].to(
                    device
                )

                logits = relation_batch(
                    head,
                    hidden,
                    pair_indices,
                    geom,
                    page["pixel_features"][chunk],
                    device,
                )

                loss = loss_fn(
                    logits,
                    y,
                )

                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    head.parameters(),
                    1.0,
                )

                optimizer.step()

                epoch_loss += float(
                    loss.item()
                )

                epoch_steps += 1

                epoch_pairs += len(
                    chunk
                )

                epoch_pos += int(
                    y.sum().item()
                )

        train_loss = (
            epoch_loss /
            max(
                1,
                epoch_steps,
            )
        )

        val_metrics = evaluate_pages_raw(
            head,
            val_pages,
            device,
        )

        print()
        print(
            "[EPOCH {:02d}] train_loss={:.4f} "
            "pairs={} pos={} "
            "val_F1={:.4f} "
            "val_P={:.4f} "
            "val_R={:.4f} "
            "val_AUC={:.4f} "
            "val_AP={:.4f} "
            "best_thr={:.2f}".format(
                epoch,
                train_loss,
                epoch_pairs,
                epoch_pos,
                val_metrics["f1"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["auc"],
                val_metrics["ap"],
                val_metrics["threshold"],
            )
        )

        # AP is the main checkpoint criterion.
        if val_metrics[
            "ap"
        ] > best_ap:

            best_ap = val_metrics[
                "ap"
            ]

            save_head(
                output_path,
                head,
                config,
            )

    print()
    print(
        "[DONE] best validation AP={:.4f}".format(
            best_ap
        )
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The resulting head is a NEW V4 pixel-aware word-level head."
    )

    print(
        "It is NOT compatible with the old segment_head.pt."
    )


# =============================================================================
# Diagnose command
# =============================================================================

def diagnose(args):

    import torch

    device = args.device

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    tokenizer, model = load_layoutlmv3(
        Path(args.model_path),
        device,
    )

    head, _ = load_head(
        Path(args.head),
        device,
    )

    image_dir = Path(
        args.images
    )

    ann_dir = Path(
        args.annotations
    )

    if args.stems:

        stems = list(
            args.stems
        )

    else:

        stems = list_stems(
            image_dir,
            ann_dir,
        )

    print()
    print(
        "=" * 60
    )
    print(
        "TARGET + CANDIDATE + RAW MLP DIAGNOSTIC"
    )
    print(
        "=" * 60
    )

    pages = []

    total_words = 0
    total_forms = 0
    total_mixed = 0
    total_runs_removed = 0

    candidate_gold_total = 0
    candidate_gold_covered = 0

    for idx, stem in enumerate(
        stems,
        start=1,
    ):

        print()
        print(
            "[PAGE {:03d}/{}] {}".format(
                idx,
                len(stems),
                stem,
            )
        )

        annotation_path = (
            ann_dir /
            (
                stem +
                ".json"
            )
        )

        image_path = (
            image_dir /
            (
                stem +
                ".png"
            )
        )

        words, entities = load_page_annotation(
            annotation_path
        )

        page_box = union_boxes(
            w.box
            for w in words
        )

        # V4 diagnostic must build the pixel context for this page before
        # generating pixel-aware candidates.  The previous generated file
        # accidentally referenced a stale/nonexistent variable here.
        image = Image.open(
            str(image_path)
        ).convert("RGB")

        pixel_context = build_pixel_context(
            image,
            page_box,
        )

        candidates = candidate_pairs(
            words,
            page_box,
            top_k=args.top_k,
            pixel_context=pixel_context,
            pixel_top_k=args.pixel_top_k,
        )

        try:
            image.close()
        except Exception:
            pass

        cr = candidate_recall(
            words,
            candidates,
        )

        candidate_gold_total += cr[
            "gold_positive_pairs"
        ]

        candidate_gold_covered += cr[
            "covered_positive_pairs"
        ]

        print(
            "[TARGET] native FUNSD form.id"
        )

        print(
            "[TARGET] words={} forms={}".format(
                len(words),
                len(
                    set(
                        w.segment_id
                        for w in words
                        if w.segment_id is not None
                    )
                ),
            )
        )

        print(
            "[CAND] pairs={} gold_pos={} covered={} recall={:.4f}".format(
                len(candidates),
                cr[
                    "gold_positive_pairs"
                ],
                cr[
                    "covered_positive_pairs"
                ],
                cr[
                    "recall"
                ],
            )
        )

        page = prepare_page(
            tokenizer,
            model,
            image_path,
            annotation_path,
            device,
            args.max_length,
            use_gold_candidate_injection=False,
            top_k=args.top_k,
        )

        page["stem"] = stem

        pages.append(
            page
        )

        total_words += len(
            words
        )

        total_forms += len(
            entities
        )

    # -------------------------------------------------------------------------
    # Summary.
    # -------------------------------------------------------------------------

    print()
    print(
        "=" * 60
    )
    print(
        "TARGET SUMMARY"
    )
    print(
        "=" * 60
    )

    print(
        "Pages evaluated                 : {}".format(
            len(pages)
        )
    )

    print(
        "Total words                     : {}".format(
            total_words
        )
    )

    print(
        "Total FUNSD form items          : {}".format(
            total_forms
        )
    )

    print(
        "Gold target                     : form.id"
    )

    print(
        "Custom segment IDs              : none"
    )

    print(
        "Linking used as segment label   : NO"
    )

    candidate_recall_value = safe_div(
        candidate_gold_covered,
        candidate_gold_total,
    )

    print()
    print(
        "=" * 60
    )
    print(
        "CANDIDATE SUMMARY"
    )
    print(
        "=" * 60
    )

    print(
        "Gold positive pairs             : {}".format(
            candidate_gold_total
        )
    )

    print(
        "Covered by candidates           : {}".format(
            candidate_gold_covered
        )
    )

    print(
        "Candidate positive recall       : {:.4f}".format(
            candidate_recall_value
        )
    )

    if candidate_recall_value < 0.90:

        print()
        print(
            "WARNING: candidate recall < 0.90"
        )

        print(
            "The candidate graph should still be broadened."
        )

    else:

        print()
        print(
            "Candidate recall is acceptable."
        )

    raw_summary = raw_mlp_diagnosis(
        head,
        pages,
        device,
    )

    # -------------------------------------------------------------------------
    # Save JSON.
    # -------------------------------------------------------------------------

    output = Path(
        args.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "pages": len(
            pages
        ),
        "total_words": total_words,
        "total_forms": total_forms,
        "target": "FUNSD form.id",
        "candidate_gold_positive_pairs":
            candidate_gold_total,
        "candidate_gold_positive_covered":
            candidate_gold_covered,
        "candidate_recall":
            candidate_recall_value,
        "raw_mlp":
            raw_summary,
    }

    with open(
        str(
            output /
            "diagnostic_summary.json"
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "[OUT] {}".format(
            (
                output /
                "diagnostic_summary.json"
            ).resolve()
        )
    )


# =============================================================================
# Predict
# =============================================================================

def predict(args):

    import torch

    device = args.device

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    tokenizer, model = load_layoutlmv3(
        Path(args.model_path),
        device,
    )

    head, _ = load_head(
        Path(args.head),
        device,
    )

    image_dir = Path(
        args.images
    )

    ann_dir = Path(
        args.annotations
    )

    if args.stems:

        stems = list(
            args.stems
        )

    else:

        stems = list_stems(
            image_dir,
            ann_dir,
        )

    output_root = Path(
        args.output
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {}

    macro_p = []
    macro_r = []
    macro_f1 = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for idx, stem in enumerate(
        stems,
        start=1,
    ):

        print()
        print(
            "=" * 60
        )

        print(
            "[PAGE {:03d}/{}] {}".format(
                idx,
                len(stems),
                stem,
            )
        )

        print(
            "=" * 60
        )

        image_path = (
            image_dir /
            (
                stem +
                ".png"
            )
        )

        ann_path = (
            ann_dir /
            (
                stem +
                ".json"
            )
        )

        image = Image.open(
            str(image_path)
        ).convert("RGB")

        words, entities = load_page_annotation(
            ann_path
        )

        page_box = union_boxes(
            w.box
            for w in words
        )

        pixel_context = build_pixel_context(
            image,
            page_box,
        )

        extract_word_hidden(
            tokenizer,
            model,
            image,
            words,
            page_box,
            device,
            args.max_length,
        )

        candidates = candidate_pairs(
            words,
            page_box,
            top_k=args.top_k,
            pixel_context=pixel_context,
            pixel_top_k=args.pixel_top_k,
        )

        geometry = make_pair_geometry(
            words,
            candidates,
            page_box,
        )

        pixel_features = make_pair_pixel_features(
            words,
            candidates,
            pixel_context,
        )

        hidden = torch_stack_hidden(
            words,
            words[0].hidden.shape[-1],
        )

        probs = predict_pairs(
            head,
            hidden,
            candidates,
            geometry,
            pixel_features,
            device,
        )

        groups = cluster_words(
            words,
            candidates,
            probs,
            pixel_features,
            page_box,
            threshold=args.threshold,
            top_k=args.edge_top_k,
            pixel_gate=args.pixel_gate,
            pixel_order_weight=args.pixel_order_weight,
        )

        segments = groups_to_segments(
            groups,
            words,
        )

        metrics = None

        if args.evaluate:

            metrics = pairwise_segment_metrics(
                words,
                segments,
            )

            print(
                "[EVAL] segments={} "
                "P={:.4f} "
                "R={:.4f} "
                "F1={:.4f}".format(
                    metrics[
                        "num_pred_segments"
                    ],
                    metrics[
                        "pairwise_precision"
                    ],
                    metrics[
                        "pairwise_recall"
                    ],
                    metrics[
                        "pairwise_f1"
                    ],
                )
            )

            macro_p.append(
                metrics[
                    "pairwise_precision"
                ]
            )

            macro_r.append(
                metrics[
                    "pairwise_recall"
                ]
            )

            macro_f1.append(
                metrics[
                    "pairwise_f1"
                ]
            )

            total_tp += metrics[
                "tp"
            ]

            total_fp += metrics[
                "fp"
            ]

            total_fn += metrics[
                "fn"
            ]

            total_tn += metrics[
                "tn"
            ]

        page_out = (
            output_root /
            stem
        )

        page_out.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            str(
                page_out /
                "segments.json"
            ),
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "stem": stem,
                    "num_words": len(
                        words
                    ),
                    "num_candidates": len(
                        candidates
                    ),
                    "num_segments": len(
                        segments
                    ),
                    "threshold": args.threshold,
                    "segments": segments,
                    "metrics": metrics,
                    "top_pair_probabilities": [
                        {
                            "a": pair[0],
                            "b": pair[1],
                            "p_same": p,
                        }
                        for pair, p in sorted(
                            zip(
                                candidates,
                                probs,
                            ),
                            key=lambda x:
                            x[1],
                            reverse=True,
                        )[:500]
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        visualize_page(
            image,
            words,
            segments,
            page_out /
            "visualization.png",
            "{} | word-level V4 pixel".format(
                stem
            ),
            metrics,
        )

        result[stem] = {
            "num_segments": len(
                segments
            ),
            "metrics": metrics,
        }

        try:
            image.close()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Full split summary.
    # -------------------------------------------------------------------------

    if macro_f1:

        macro_precision = (
            sum(macro_p)
            /
            len(macro_p)
        )

        macro_recall = (
            sum(macro_r)
            /
            len(macro_r)
        )

        macro_f1_value = (
            sum(macro_f1)
            /
            len(macro_f1)
        )

        micro_precision = safe_div(
            total_tp,
            total_tp +
            total_fp,
        )

        micro_recall = safe_div(
            total_tp,
            total_tp +
            total_fn,
        )

        micro_f1 = safe_div(
            2.0 *
            micro_precision *
            micro_recall,
            micro_precision +
            micro_recall,
        )

        print()
        print(
            "=" * 60
        )

        print(
            "FULL SPLIT SUMMARY"
        )

        print(
            "=" * 60
        )

        print(
            "Pages evaluated : {}".format(
                len(
                    macro_f1
                )
            )
        )

        print(
            "MACRO P={:.4f} R={:.4f} F1={:.4f}".format(
                macro_precision,
                macro_recall,
                macro_f1_value,
            )
        )

        print(
            "MICRO P={:.4f} R={:.4f} F1={:.4f}".format(
                micro_precision,
                micro_recall,
                micro_f1,
            )
        )

        print(
            "Threshold       : {:.3f}".format(
                args.threshold
            )
        )

        print(
            "Edge top-k      : {}".format(
                args.edge_top_k
            )
        )

    with open(
        str(
            output_root /
            "metrics.json"
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "[OUT] {}".format(
            output_root.resolve()
        )
    )


# =============================================================================
# CLI
# =============================================================================

def cuda_available():

    try:

        import torch

        return bool(
            torch.cuda.is_available()
        )

    except Exception:

        return False


def add_model_args(parser):

    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if cuda_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_TEXT_LEN,
    )


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "FUNSD word-level segment grouping V4 pixel-aware"
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # -------------------------------------------------------------------------
    # train
    # -------------------------------------------------------------------------

    tr = sub.add_parser(
        "train"
    )

    add_model_args(
        tr
    )

    tr.add_argument(
        "--train-images",
        default=DEFAULT_TRAIN_IMAGES,
    )

    tr.add_argument(
        "--train-annotations",
        default=DEFAULT_TRAIN_ANN,
    )

    tr.add_argument(
        "--val-images",
        default=None,
    )

    tr.add_argument(
        "--val-annotations",
        default=None,
    )

    tr.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
    )

    tr.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    tr.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    tr.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
    )

    tr.add_argument(
        "--projection-size",
        type=int,
        default=HIDDEN_PROJECTION,
    )

    tr.add_argument(
        "--head-width",
        type=int,
        default=256,
    )

    tr.add_argument(
        "--head-bottleneck",
        type=int,
        default=64,
    )

    tr.add_argument(
        "--dropout",
        type=float,
        default=0.10,
    )

    tr.add_argument(
        "--negative-ratio",
        type=int,
        default=3,
    )

    tr.add_argument(
        "--max-positive-per-page",
        type=int,
        default=4000,
    )

    tr.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    tr.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    tr.add_argument(
        "--pixel-top-k",
        type=int,
        default=24,
    )

    tr.add_argument(
        "--head-out",
        default=(
            DEFAULT_OUT +
            "/segment_head_v4_pixel.pt"
        ),
    )

    tr.add_argument(
        "--seed",
        type=int,
        default=1993,
    )

    # -------------------------------------------------------------------------
    # diagnose
    # -------------------------------------------------------------------------

    dg = sub.add_parser(
        "diagnose"
    )

    add_model_args(
        dg
    )

    dg.add_argument(
        "--head",
        required=True,
    )

    dg.add_argument(
        "--images",
        default=DEFAULT_TEST_IMAGES,
    )

    dg.add_argument(
        "--annotations",
        default=DEFAULT_TEST_ANN,
    )

    dg.add_argument(
        "--stems",
        nargs="*",
        default=None,
    )

    dg.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    dg.add_argument(
        "--pixel-top-k",
        type=int,
        default=24,
    )

    dg.add_argument(
        "--output",
        default=(
            DEFAULT_OUT +
            "/diagnostic_v4_pixel"
        ),
    )

    # -------------------------------------------------------------------------
    # predict
    # -------------------------------------------------------------------------

    pr = sub.add_parser(
        "predict"
    )

    add_model_args(
        pr
    )

    pr.add_argument(
        "--head",
        required=True,
    )

    pr.add_argument(
        "--images",
        default=DEFAULT_TEST_IMAGES,
    )

    pr.add_argument(
        "--annotations",
        default=DEFAULT_TEST_ANN,
    )

    pr.add_argument(
        "--stems",
        nargs="*",
        default=None,
    )

    pr.add_argument(
        "--threshold",
        type=float,
        default=0.50,
    )

    pr.add_argument(
        "--edge-top-k",
        type=int,
        default=8,
    )

    pr.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    pr.add_argument(
        "--pixel-top-k",
        type=int,
        default=24,
    )

    pr.add_argument(
        "--pixel-gate",
        type=float,
        default=0.12,
    )

    pr.add_argument(
        "--pixel-order-weight",
        type=float,
        default=0.10,
    )

    pr.add_argument(
        "--evaluate",
        action="store_true",
    )

    pr.add_argument(
        "--output",
        default=(
            DEFAULT_OUT +
            "/test_v4_pixel"
        ),
    )

    return parser


def main():

    parser = build_parser()

    args = parser.parse_args()

    if args.command == "train":

        train(args)

    elif args.command == "diagnose":

        diagnose(args)

    elif args.command == "predict":

        predict(args)

    else:

        raise ValueError(
            args.command
        )


if __name__ == "__main__":
    main()

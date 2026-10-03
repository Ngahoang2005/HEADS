#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric

import transformers
import torch

from layoutlmft.data import DataCollatorForKeyValueExtraction
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from layoutlmft.data.image_utils import (
    RandomResizedCropAndInterpolationWithTwoPic,
    pil_loader,
    Compose,
)

from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from torchvision import transforms


check_min_version("4.5.0")
logger = logging.getLogger(__name__)


# ==============================================================================
# ARGUMENTS
# ==============================================================================

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)

    use_hierarchical_position_encoding: bool = field(default=False)
    max_line_position: int = field(default=50)
    max_block_position: int = field(default=20)
    use_column_encoding: bool = field(default=False)
    max_column_position: int = field(default=10)

    use_intra_line_boundary: bool = field(default=False)
    lambda_bound_init: float = field(default=0.1)

    use_semantic_geometry_disentangle: bool = field(default=False)
    lambda_geo_init: float = field(default=0.1)
    lambda_orth_init: float = field(default=0.1)


@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(default="ner")
    dataset_name: Optional[str] = field(default="funsd")
    dataset_config_name: Optional[str] = field(default=None)

    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)

    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    pad_to_max_length: bool = field(default=True)

    max_train_samples: Optional[int] = field(default=None)
    max_val_samples: Optional[int] = field(default=None)
    max_test_samples: Optional[int] = field(default=None)

    label_all_tokens: bool = field(default=False)
    return_entity_level_metrics: bool = field(default=False)

    segment_level_layout: bool = field(default=True)
    visual_embed: bool = field(default=True)
    use_segment_head: bool = field(default=False)

    # Automatic geometry-only reading-order / segmentation.
    # There is intentionally NO oracle-segment path in this file.
    apply_xy_cut: bool = field(
        default=True,
        metadata={"help": "Build reading order from OCR bounding boxes using recursive XY-Cut."},
    )

    data_dir: Optional[str] = field(default=None)

    input_size: int = field(default=224)
    second_input_size: int = field(default=112)

    train_interpolation: str = field(default="bicubic")
    second_interpolation: str = field(default="lanczos")
    imagenet_default_mean_and_std: bool = field(default=False)


# ==============================================================================
# GEOMETRIC PRIMITIVES
# ==============================================================================

def _height(box):
    return max(1.0, float(box[3] - box[1]))


def _width(box):
    return max(1.0, float(box[2] - box[0]))


def _x_center(box):
    return 0.5 * (float(box[0]) + float(box[2]))


def _y_center(box):
    return 0.5 * (float(box[1]) + float(box[3]))


def _horizontal_gap(box_a, box_b):
    if float(box_a[2]) < float(box_b[0]):
        return float(box_b[0]) - float(box_a[2])
    if float(box_b[2]) < float(box_a[0]):
        return float(box_a[0]) - float(box_b[2])
    return 0.0


def _vertical_gap(box_a, box_b):
    if float(box_a[3]) < float(box_b[1]):
        return float(box_b[1]) - float(box_a[3])
    if float(box_b[3]) < float(box_a[1]):
        return float(box_a[1]) - float(box_b[3])
    return 0.0


def _vertical_overlap_ratio(box_a, box_b):
    top = max(float(box_a[1]), float(box_b[1]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, bottom - top)
    denom = max(1.0, min(_height(box_a), _height(box_b)))
    return inter / denom


def _horizontal_overlap_ratio(box_a, box_b):
    left = max(float(box_a[0]), float(box_b[0]))
    right = min(float(box_a[2]), float(box_b[2]))
    inter = max(0.0, right - left)
    denom = max(1.0, min(_width(box_a), _width(box_b)))
    return inter / denom


# ==============================================================================
# STEP 1: VISUAL LINE CONSTRUCTION
# ==============================================================================

def build_visual_lines(bboxes):
    """
    Build spatial text lines from OCR word boxes.

    IMPORTANT:
      - Uses only bounding boxes.
      - Does not inspect labels.
      - Does not use dataset-provided/oracle segments.

    A line is formed from words that are vertically aligned and close enough
    horizontally to plausibly belong to one visual text line.
    """
    n = len(bboxes)
    if n == 0:
        return []

    heights = np.asarray([_height(b) for b in bboxes], dtype=np.float32)
    median_h = max(1.0, float(np.median(heights)))

    # Adaptive thresholds.
    y_center_tol = 0.65 * median_h
    max_x_gap = 4.0 * median_h

    # Start from top-to-bottom, then left-to-right.
    order = sorted(
        range(n),
        key=lambda i: (_y_center(bboxes[i]), float(bboxes[i][0]), i),
    )

    lines = []

    for idx in order:
        box = bboxes[idx]
        yc = _y_center(box)

        best = None
        best_score = None

        for li, line in enumerate(lines):
            line_box = line["bbox"]

            dy = abs(yc - line["y_center"])
            overlap = _vertical_overlap_ratio(box, line_box)
            x_gap = _horizontal_gap(box, line_box)

            # Reject clearly different rows.
            if overlap < 0.20 and dy > y_center_tol:
                continue

            # Reject distant columns.
            if x_gap > max_x_gap:
                continue

            score = (
                dy / median_h
                + 0.20 * (x_gap / max_x_gap)
                - 1.50 * overlap
            )

            if best_score is None or score < best_score:
                best_score = score
                best = li

        if best is None:
            lines.append(
                {
                    "indices": [idx],
                    "bbox": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ],
                    "y_center": yc,
                }
            )
        else:
            line = lines[best]
            line["indices"].append(idx)
            line["bbox"] = [
                min(line["bbox"][0], float(box[0])),
                min(line["bbox"][1], float(box[1])),
                max(line["bbox"][2], float(box[2])),
                max(line["bbox"][3], float(box[3])),
            ]
            line["y_center"] = float(
                np.mean([_y_center(bboxes[j]) for j in line["indices"]])
            )

    # Always enforce left-to-right order inside each visual line.
    for line in lines:
        line["indices"].sort(
            key=lambda i: (float(bboxes[i][0]), _y_center(bboxes[i]), i)
        )

    # Temporary top-to-bottom ordering for deterministic region construction.
    lines.sort(
        key=lambda line: (
            float(line["bbox"][1]),
            float(line["bbox"][0]),
        )
    )

    return lines


# ==============================================================================
# STEP 2: XY-CUT ON LINE REGIONS (NOT ON INDIVIDUAL WORDS)
# ==============================================================================

def _largest_gap_split(regions, axis, median_h):
    """
    Find the largest meaningful whitespace split.

    axis=0 -> vertical partition (left/right), searching x gaps.
    axis=1 -> horizontal partition (top/bottom), searching y gaps.
    """
    if len(regions) <= 1:
        return None

    start = 0 if axis == 0 else 1
    end = 2 if axis == 0 else 3

    ordered = sorted(
        regions,
        key=lambda r: (
            float(r["bbox"][start]),
            float(r["bbox"][end]),
        ),
    )

    best_gap = -1.0
    best_index = -1
    running_end = float(ordered[0]["bbox"][end])

    for i in range(1, len(ordered)):
        current_start = float(ordered[i]["bbox"][start])
        gap = current_start - running_end

        if gap > best_gap:
            best_gap = gap
            best_index = i

        running_end = max(
            running_end,
            float(ordered[i]["bbox"][end]),
        )

    if best_index <= 0 or best_index >= len(ordered):
        return None

    # Normalise by text height so the algorithm is scale-robust.
    gap_ratio = best_gap / median_h

    # Vertical whitespace between adjacent text lines is usually small.
    # Column gaps are typically much wider.
    if axis == 0:
        min_ratio = 1.50
    else:
        min_ratio = 0.85

    if gap_ratio < min_ratio:
        return None

    return {
        "regions_a": ordered[:best_index],
        "regions_b": ordered[best_index:],
        "gap": best_gap,
        "score": gap_ratio,
        "axis": axis,
    }


def xy_cut_reading_order(lines, median_h):
    """
    Recursive XY-Cut over visual-line bounding regions.

    The recursion first looks for a genuine whitespace partition. At each
    node both x- and y-splits are considered; the strongest normalized gap is
    selected. Leaves are emitted top-to-bottom and left-to-right.
    """
    if len(lines) <= 1:
        return list(lines)

    candidates = []

    x_split = _largest_gap_split(
        lines,
        axis=0,
        median_h=median_h,
    )
    y_split = _largest_gap_split(
        lines,
        axis=1,
        median_h=median_h,
    )

    if x_split is not None:
        # Give a modest preference to a strong column gap.
        x_split["score"] *= 1.10
        candidates.append(x_split)

    if y_split is not None:
        candidates.append(y_split)

    if not candidates:
        return sorted(
            lines,
            key=lambda line: (
                float(line["bbox"][1]),
                float(line["bbox"][0]),
            ),
        )

    split = max(candidates, key=lambda x: x["score"])

    left = xy_cut_reading_order(
        split["regions_a"],
        median_h,
    )
    right = xy_cut_reading_order(
        split["regions_b"],
        median_h,
    )

    return left + right


# ==============================================================================
# STEP 3: BUILD AUTO SEGMENTS FROM THE READING-ORDERED VISUAL LINES
# ==============================================================================

def build_reading_order_and_segments(bboxes):
    """
    Geometry-only pipeline:

        word boxes
            -> visual lines
            -> XY-Cut reading order
            -> one visual line = one heuristic segment

    Returns:
        order: word indices in reading order
        seg_ids_orig: segment ID for every original word index

    No dataset-provided segment information is used.
    """
    if not bboxes:
        return [], []

    lines = build_visual_lines(bboxes)

    if not lines:
        return list(range(len(bboxes))), list(range(len(bboxes)))

    median_h = max(
        1.0,
        float(np.median([line["bbox"][3] - line["bbox"][1] for line in lines])),
    )

    ordered_lines = xy_cut_reading_order(
        lines,
        median_h,
    )

    # Flatten visual lines into the final reading order.
    order = []

    # Segment ID is defined by the ordered visual line.
    seg_ids_orig = [-1] * len(bboxes)

    for seg_id, line in enumerate(ordered_lines):
        for idx in line["indices"]:
            order.append(idx)
            seg_ids_orig[idx] = seg_id

    # Defensive consistency check.
    if len(order) != len(bboxes) or len(set(order)) != len(bboxes):
        order = list(range(len(bboxes)))

    for i, sid in enumerate(seg_ids_orig):
        if sid < 0:
            seg_ids_orig[i] = i

    return order, seg_ids_orig


# ==============================================================================
# OPTIONAL HIERARCHICAL POSITION IDS
# ==============================================================================

def compute_line_ids(bboxes):
    """
    Compute line IDs from the already reading-ordered boxes.

    IDs follow the same visual-line construction used for segmentation.
    """
    if not bboxes:
        return []

    lines = build_visual_lines(bboxes)
    line_map = [-1] * len(bboxes)

    # Sort by first occurrence in the current reading-order list.
    lines.sort(key=lambda line: min(line["indices"]))

    for lid, line in enumerate(lines):
        for idx in line["indices"]:
            line_map[idx] = lid

    for i in range(len(line_map)):
        if line_map[i] < 0:
            line_map[i] = i

    return line_map


def compute_block_ids(bboxes, x_threshold=50, y_threshold=30):
    """
    Conservative spatial block IDs on reading-ordered boxes.
    """
    if not bboxes:
        return []

    centers = [
        (_x_center(box), _y_center(box))
        for box in bboxes
    ]

    blocks = [0]
    current_block = 0

    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i - 1][0]
        dy = centers[i][1] - centers[i - 1][1]

        if abs(dx) > x_threshold or abs(dy) > y_threshold:
            current_block += 1

        blocks.append(current_block)

    return blocks


def compute_column_ids(bboxes, x_threshold=50):
    """
    Column IDs estimated from x-centers in reading order.
    """
    if not bboxes:
        return []

    x_centers = [_x_center(box) for box in bboxes]
    columns = [0]
    current_col = 0

    for i in range(1, len(x_centers)):
        if abs(x_centers[i] - x_centers[i - 1]) > x_threshold:
            current_col += 1

        columns.append(current_col)

    return columns


def compute_entity_ids(label_ids_aligned, label_list):
    """
    Assign a unique ID to each contiguous entity for downstream models
    expecting entity_ids.
    """
    entity_ids = []
    current_id = -1
    prev_type = None

    for lid in label_ids_aligned:
        if lid == -100:
            entity_ids.append(-1)
            continue

        label_str = label_list[lid]

        if label_str == "O":
            entity_ids.append(-1)
            prev_type = None
            continue

        prefix, entity_type = label_str.split("-", 1)

        if prefix == "B" or entity_type != prev_type:
            current_id += 1

        entity_ids.append(current_id)
        prev_type = entity_type

    return entity_ids


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --------------------------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------------------------
    last_checkpoint = None

    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(
            training_args.output_dir
        )

        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. "
                "To avoid this behavior, change the output_dir or add "
                "--overwrite_output_dir to train from scratch."
            )

    # --------------------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------------------
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(
        logging.INFO
        if is_main_process(training_args.local_rank)
        else logging.WARN
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    logger.info(
        f"Training/evaluation parameters {training_args}"
    )

    # --------------------------------------------------------------------------
    # Seed
    # --------------------------------------------------------------------------
    set_seed(training_args.seed)

    # --------------------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------------------
    if data_args.dataset_name == "funsd":
        import layoutlmft.data.funsd

        datasets = load_dataset(
            os.path.abspath(layoutlmft.data.funsd.__file__),
            cache_dir=model_args.cache_dir,
        )

    elif data_args.dataset_name == "cord":
        import layoutlmft.data.cord

        datasets = load_dataset(
            os.path.abspath(layoutlmft.data.cord.__file__),
            cache_dir=model_args.cache_dir,
        )

    else:
        raise NotImplementedError(
            f"Unsupported dataset_name={data_args.dataset_name}. "
            "This runner supports FUNSD and CORD."
        )

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["test"].column_names
        features = datasets["test"].features

    text_column_name = (
        "words"
        if "words" in column_names
        else "tokens"
    )

    label_column_name = (
        f"{data_args.task_name}_tags"
        if f"{data_args.task_name}_tags" in column_names
        else column_names[1]
    )

    remove_columns = column_names

    if isinstance(
        features[label_column_name].feature,
        ClassLabel,
    ):
        label_list = features[label_column_name].feature.names
        label_to_id = {
            i: i
            for i in range(len(label_list))
        }
    else:
        unique_labels = set()

        for label_seq in datasets["train"][label_column_name]:
            unique_labels = unique_labels | set(label_seq)

        label_list = sorted(list(unique_labels))
        label_to_id = {
            label: i
            for i, label in enumerate(label_list)
        }

    num_labels = len(label_list)

    # --------------------------------------------------------------------------
    # Model config
    # --------------------------------------------------------------------------
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        input_size=data_args.input_size,
        use_auth_token=(
            True if model_args.use_auth_token else None
        ),
        use_hierarchical_position_encoding=(
            model_args.use_hierarchical_position_encoding
        ),
        max_line_position=model_args.max_line_position,
        max_block_position=model_args.max_block_position,
        use_column_encoding=model_args.use_column_encoding,
        max_column_position=model_args.max_column_position,
        use_intra_line_boundary=model_args.use_intra_line_boundary,
        lambda_bound_init=model_args.lambda_bound_init,
        use_semantic_geometry_disentangle=(
            model_args.use_semantic_geometry_disentangle
        ),
        lambda_geo_init=model_args.lambda_geo_init,
        lambda_orth_init=model_args.lambda_orth_init,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        tokenizer_file=None,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
        revision=model_args.model_revision,
        use_auth_token=(
            True if model_args.use_auth_token else None
        ),
    )

    if getattr(data_args, "use_segment_head", False):
        from layoutlmft.models.layoutlmv3.modeling_layoutlmv3_segment import (
            LayoutLMv3ForSegmentTokenClassification,
        )

        model = (
            LayoutLMv3ForSegmentTokenClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(
                    ".ckpt" in model_args.model_name_or_path
                ),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=(
                    True if model_args.use_auth_token else None
                ),
            )
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(
                ".ckpt" in model_args.model_name_or_path
            ),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=(
                True if model_args.use_auth_token else None
            ),
        )

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer."
        )

    # --------------------------------------------------------------------------
    # Visual preprocessing
    # --------------------------------------------------------------------------
    padding = (
        "max_length"
        if data_args.pad_to_max_length
        else False
    )

    if data_args.visual_embed:
        imagenet_default_mean_and_std = (
            data_args.imagenet_default_mean_and_std
        )

        mean = (
            IMAGENET_INCEPTION_MEAN
            if not imagenet_default_mean_and_std
            else IMAGENET_DEFAULT_MEAN
        )

        std = (
            IMAGENET_INCEPTION_STD
            if not imagenet_default_mean_and_std
            else IMAGENET_DEFAULT_STD
        )

        common_transform = Compose(
            [
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=data_args.input_size,
                    interpolation=data_args.train_interpolation,
                ),
            ]
        )

        patch_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std),
                ),
            ]
        )

    # --------------------------------------------------------------------------
    # Tokenization + geometry-only reading order + auto segmentation
    # --------------------------------------------------------------------------
    def tokenize_and_align_labels(examples, augmentation=False):
        # Build the geometry-only reading order BEFORE tokenization.
        reordered_words = []
        reordered_bboxes = []
        reordered_labels = []
        reordered_orders = []

        for sample_idx in range(
            len(examples[text_column_name])
        ):
            words_i = examples[text_column_name][sample_idx]
            bboxes_i = examples["bboxes"][sample_idx]
            labels_i = examples[label_column_name][sample_idx]

            if (
                getattr(data_args, "apply_xy_cut", True)
                and len(bboxes_i) > 1
            ):
                order, _ = build_reading_order_and_segments(
                    bboxes_i
                )
            else:
                order = list(range(len(bboxes_i)))

            reordered_orders.append(order)

            reordered_words.append(
                [words_i[j] for j in order]
            )
            reordered_bboxes.append(
                [bboxes_i[j] for j in order]
            )
            reordered_labels.append(
                [labels_i[j] for j in order]
            )

        tokenized_inputs = tokenizer(
            reordered_words,
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,
            is_split_into_words=True,
        )

        labels = []
        bboxes = []
        images = []

        seg_ids = []
        line_ids_all = []
        block_ids_all = []
        column_ids_all = []
        entity_ids_all = []

        for batch_index in range(
            len(tokenized_inputs["input_ids"])
        ):
            word_ids = tokenized_inputs.word_ids(
                batch_index=batch_index
            )

            original_batch_index = (
                tokenized_inputs[
                    "overflow_to_sample_mapping"
                ][batch_index]
            )

            label = reordered_labels[
                original_batch_index
            ]
            bbox = reordered_bboxes[
                original_batch_index
            ]
            words = reordered_words[
                original_batch_index
            ]

            # --------------------------------------------------------------
            # AUTO SEGMENTS: rebuild from the reordered word boxes.
            #
            # Absolutely no dataset-provided segment IDs are read here.
            # --------------------------------------------------------------
            if getattr(
                data_args,
                "use_segment_head",
                False,
            ):
                _, word_seg_id = (
                    build_reading_order_and_segments(
                        bbox
                    )
                )
            else:
                word_seg_id = None

            line_ids_orig = compute_line_ids(bbox)
            block_ids_orig = compute_block_ids(bbox)
            column_ids_orig = compute_column_ids(bbox)

            previous_word_idx = None

            label_ids = []
            bbox_inputs = []
            seg_id_inputs = []
            line_ids_aligned = []
            block_ids_aligned = []
            column_ids_aligned = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append(
                        [0, 0, 0, 0]
                    )

                    if word_seg_id is not None:
                        seg_id_inputs.append(-1)

                    line_ids_aligned.append(-1)
                    block_ids_aligned.append(-1)
                    column_ids_aligned.append(-1)

                elif word_idx != previous_word_idx:
                    label_ids.append(
                        label_to_id[
                            label[word_idx]
                        ]
                    )

                    bbox_inputs.append(
                        bbox[word_idx]
                    )

                    if word_seg_id is not None:
                        seg_id_inputs.append(
                            word_seg_id[word_idx]
                        )

                    line_ids_aligned.append(
                        line_ids_orig[word_idx]
                    )
                    block_ids_aligned.append(
                        block_ids_orig[word_idx]
                    )
                    column_ids_aligned.append(
                        column_ids_orig[word_idx]
                    )

                else:
                    label_ids.append(
                        label_to_id[
                            label[word_idx]
                        ]
                        if data_args.label_all_tokens
                        else -100
                    )

                    bbox_inputs.append(
                        bbox[word_idx]
                    )

                    if word_seg_id is not None:
                        seg_id_inputs.append(
                            word_seg_id[word_idx]
                        )

                    line_ids_aligned.append(
                        line_ids_orig[word_idx]
                    )
                    block_ids_aligned.append(
                        block_ids_orig[word_idx]
                    )
                    column_ids_aligned.append(
                        column_ids_orig[word_idx]
                    )

                previous_word_idx = word_idx

            labels.append(label_ids)
            bboxes.append(bbox_inputs)

            if word_seg_id is not None:
                seg_ids.append(seg_id_inputs)

            line_ids_all.append(line_ids_aligned)
            block_ids_all.append(block_ids_aligned)
            column_ids_all.append(column_ids_aligned)

            entity_ids_all.append(
                compute_entity_ids(
                    label_ids,
                    label_list,
                )
            )

            if data_args.visual_embed:
                ipath = examples["image_path"][
                    original_batch_index
                ]

                img = pil_loader(ipath)

                for_patches, _ = common_transform(
                    img,
                    augmentation=augmentation,
                )

                patch = patch_transform(
                    for_patches
                )

                images.append(patch)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["line_ids"] = line_ids_all
        tokenized_inputs["block_ids"] = block_ids_all
        tokenized_inputs["column_ids"] = column_ids_all
        tokenized_inputs["entity_ids"] = entity_ids_all

        if getattr(
            data_args,
            "use_segment_head",
            False,
        ):
            tokenized_inputs["seg_id"] = seg_ids

        if data_args.visual_embed:
            tokenized_inputs["images"] = images

        return tokenized_inputs

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["test"].column_names
        features = datasets["test"].features

    text_column_name = "words" if "words" in column_names else "tokens"

    label_column_name = (
        f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1]
    )

    remove_columns = column_names

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        input_size=data_args.input_size,
        use_auth_token=True if model_args.use_auth_token else None,
        use_hierarchical_position_encoding=model_args.use_hierarchical_position_encoding,
        max_line_position=model_args.max_line_position,
        max_block_position=model_args.max_block_position,
        use_column_encoding=model_args.use_column_encoding,
        max_column_position=model_args.max_column_position,
        use_intra_line_boundary=model_args.use_intra_line_boundary,
        lambda_bound_init=model_args.lambda_bound_init,
        use_semantic_geometry_disentangle=model_args.use_semantic_geometry_disentangle,
        lambda_geo_init=model_args.lambda_geo_init,
        lambda_orth_init=model_args.lambda_orth_init,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        tokenizer_file=None,  # avoid loading from a cached file of the pre-trained model in another machine
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if getattr(data_args, "use_segment_head", False):
        # NEW: segment-level pooling + inter-segment context head.
        # See modeling_layoutlmv3_segment.py for the full design rationale.
        from layoutlmft.models.layoutlmv3.modeling_layoutlmv3_segment import (
    LayoutLMv3ForSegmentTokenClassification,
)
        model = LayoutLMv3ForSegmentTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    if data_args.visual_embed:
        imagenet_default_mean_and_std = data_args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        common_transform = Compose([
            # transforms.ColorJitter(0.4, 0.4, 0.4),
            # transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=data_args.input_size, interpolation=data_args.train_interpolation),
        ])
        patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples, augmentation=False):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,
            is_split_into_words=True,
        )

        labels = []
        bboxes = []
        images = []
        seg_ids = []
        line_ids_all = []    # NEW
        block_ids_all = []   # NEW
        column_ids_all = []
        entity_ids_all = []

        # Thêm vào tokenize_and_align_labels, cùng chỗ tính label_ids
        def compute_entity_ids(label_ids_aligned, label_list):
            """Gán 1 ID duy nhất cho mỗi entity liên tục, dựa trên chuỗi nhãn thật
            (không suy luận theo số chẵn/lẻ) -> tổng quát cho mọi dataset/label order."""
            entity_ids = []
            current_id = -1
            prev_type = None  # None nghĩa là đang ở "O" hoặc đầu chuỗi
            for lid in label_ids_aligned:
                if lid == -100:
                    entity_ids.append(-1)
                    continue
                label_str = label_list[lid]
                if label_str == "O":
                    entity_ids.append(-1)
                    prev_type = None
                    continue
                prefix, etype = label_str.split("-", 1)  # "B"/"I", "QUESTION"/...
                if prefix == "B" or etype != prev_type:
                    current_id += 1
                entity_ids.append(current_id)
                prev_type = etype
            return entity_ids
        
        # Helper function để tính line_ids từ bbox
        def compute_line_ids(bboxes, y_threshold=10):
            """Gom các token có y_center gần nhau thành cùng 1 dòng"""
            if not bboxes:
                return []
            # Tính y_center của mỗi bbox
            y_centers = [(box[1] + box[3]) / 2 for box in bboxes]
            # Sắp xếp và gom cụm
            lines = []
            current_line = 0
            lines.append(current_line)
            for i in range(1, len(y_centers)):
                if abs(y_centers[i] - y_centers[i-1]) > y_threshold:
                    current_line += 1
                lines.append(current_line)
            return lines
        
        # Helper function để tính block_ids từ bbox
        def compute_block_ids(bboxes, x_threshold=50, y_threshold=30):
            """Gom các token gần nhau thành cùng 1 block (dựa trên khoảng cách XY)"""
            if not bboxes:
                return []
            # Tính center của mỗi bbox
            centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in bboxes]
            # Gán block ID đơn giản: các token có khoảng cách <= threshold
            blocks = []
            current_block = 0
            blocks.append(current_block)
            for i in range(1, len(centers)):
                # Tính khoảng cách từ token hiện tại đến token trước
                dx = centers[i][0] - centers[i-1][0]
                dy = centers[i][1] - centers[i-1][1]
                if abs(dx) > x_threshold or abs(dy) > y_threshold:
                    current_block += 1
                blocks.append(current_block)
            return blocks
        def compute_column_ids(bboxes, x_threshold=50):
            """Gom các token theo cột dựa trên x_center"""
            if not bboxes:
                return []
            # Tính x_center của mỗi token
            x_centers = [(box[0] + box[2]) / 2 for box in bboxes]
            
            # Sắp xếp các token theo x_center
            columns = []
            current_col = 0
            columns.append(current_col)
            
            for i in range(1, len(x_centers)):
                # Nếu khoảng cách x lớn hơn ngưỡng → cột mới
                if abs(x_centers[i] - x_centers[i-1]) > x_threshold:
                    current_col += 1
                columns.append(current_col)
            return columns
        
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]

            # NEW: Tính line_ids và block_ids cho các token gốc
            line_ids_orig = compute_line_ids(bbox)
            block_ids_orig = compute_block_ids(bbox)
            column_ids_orig = compute_column_ids(bbox, x_threshold=50)

            # NEW: recover segment boundaries (giữ nguyên code cũ)
            word_seg_id = None
            if getattr(data_args, "use_segment_head", False):
                word_seg_id = []
                seg_counter = -1
                prev_bbox_tuple = None
                for wb in bbox:
                    wb_tuple = tuple(wb)
                    if wb_tuple != prev_bbox_tuple:
                        seg_counter += 1
                        prev_bbox_tuple = wb_tuple
                    word_seg_id.append(seg_counter)

            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            seg_id_inputs = []
            line_ids_aligned = []    # NEW
            block_ids_aligned = []   # NEW
            column_ids_aligned = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                    if word_seg_id is not None:
                        seg_id_inputs.append(-1)
                    line_ids_aligned.append(-1)     # NEW
                    block_ids_aligned.append(-1)    # NEW
                    column_ids_aligned.append(-1)
                elif word_idx != previous_word_idx:
                    # First token of a word
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                    if word_seg_id is not None:
                        seg_id_inputs.append(word_seg_id[word_idx])
                    line_ids_aligned.append(line_ids_orig[word_idx])     # NEW
                    block_ids_aligned.append(block_ids_orig[word_idx])   # NEW
                    column_ids_aligned.append(column_ids_orig[word_idx])
                else:
                    # Subsequent tokens of the same word
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                    if word_seg_id is not None:
                        seg_id_inputs.append(word_seg_id[word_idx])
                    line_ids_aligned.append(line_ids_orig[word_idx])     # NEW
                    block_ids_aligned.append(block_ids_orig[word_idx])   # NEW
                    column_ids_aligned.append(column_ids_orig[word_idx])
                previous_word_idx = word_idx
                
            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            if word_seg_id is not None:
                seg_ids.append(seg_id_inputs)
            line_ids_all.append(line_ids_aligned)     # NEW
            block_ids_all.append(block_ids_aligned)   # NEW
            column_ids_all.append(column_ids_aligned)

            entity_ids_aligned = compute_entity_ids(label_ids, label_list)  # NEW
            entity_ids_all.append(entity_ids_aligned)

            if data_args.visual_embed:
                ipath = examples["image_path"][org_batch_index]
                img = pil_loader(ipath)
                for_patches, _ = common_transform(img, augmentation=augmentation)
                patch = patch_transform(for_patches)
                images.append(patch)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["line_ids"] = line_ids_all    # NEW
        tokenized_inputs["block_ids"] = block_ids_all  # NEW
        tokenized_inputs["column_ids"] = column_ids_all
        tokenized_inputs["entity_ids"] = entity_ids_all  # NEW

        if getattr(data_args, "use_segment_head", False):
            tokenized_inputs["seg_id"] = seg_ids
        if data_args.visual_embed:
            tokenized_inputs["images"] = images

        return tokenized_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        

    if training_args.do_eval:
        validation_name = "test"
        if validation_name not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets[validation_name]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )
    # ====== KIỂM TRA BATCH DATA ======
    # Tạo data collator và dataloader để kiểm tra
    from torch.utils.data import DataLoader
    temp_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        collate_fn=data_collator,
        shuffle=False
    )
    
    # Lấy 1 batch
    batch = next(iter(temp_dataloader))
    
    # Kiểm tra các keys trong batch
    print("=" * 50)
    print("KEYS IN BATCH:", batch.keys())
    print("=" * 50)
    
    # Kiểm tra line_ids và block_ids có tồn tại không
    if "line_ids" in batch:
        print(f"✅ line_ids shape: {batch['line_ids'].shape}")
        print(f"   line_ids sample: {batch['line_ids'][0][:10]}")  # 10 token đầu
    else:
        print("❌ line_ids NOT FOUND in batch!")
    
    if "block_ids" in batch:
        print(f"✅ block_ids shape: {batch['block_ids'].shape}")
        print(f"   block_ids sample: {batch['block_ids'][0][:10]}")
    else:
        print("❌ block_ids NOT FOUND in batch!")
    
    # Kiểm tra seg_id có bị xóa không
    if "seg_id" in batch:
        print(f"✅ seg_id shape: {batch['seg_id'].shape}")
    else:
        print("⚠️ seg_id NOT FOUND (có thể bị xóa trong data_collator)")
    
    print("=" * 50)
    # ====== KẾT THÚC KIỂM TRA ======

    # Metrics
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
    # Định nghĩa Trainer tùy chỉnh để tách biệt Learning Rate
    class CustomTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                # Nhóm 1: Các tham số thuộc backbone LayoutLMv3
                backbone_params = [p for n, p in self.model.named_parameters() if "layoutlmv3" in n and p.requires_grad]
                # Nhóm 2: Các tham số mới (segment_context, classifier, is_first_token_embedding, gate)
                new_params = [p for n, p in self.model.named_parameters() if "layoutlmv3" not in n and p.requires_grad]

                optimizer_grouped_parameters = [
                    {"params": backbone_params, "lr": self.args.learning_rate}, # Dùng LR từ tham số truyền vào (VD: 1e-5)
                    {"params": new_params, "lr":5e-4} # Ép cứng LR lớn hơn cho module mới
                ]
                
                self.optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters, 
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            return self.optimizer

    # Khởi tạo Trainer bằng CustomTrainer vừa tạo thay vì Trainer mặc định
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # Initialize our Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset if training_args.do_train else None,
    #     eval_dataset=eval_dataset if training_args.do_eval else None,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    # )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

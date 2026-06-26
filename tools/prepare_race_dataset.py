#!/usr/bin/env python3
"""Prepare the competition dataset for the original FSOD-VFM pipeline.

The source dataset is kept untouched. This script:
1. validates the YOLO labels;
2. creates a deterministic, scene-grouped 80/20 train/val split;
3. writes COCO annotations for both splits;
4. samples few-shot support JSON files from the training split only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


CLASS_NAMES = [
    "HM",
    "LQS",
    "QHS",
    "MS",
    "A1_SU-35",
    "A2_C-130",
    "A3_C-17",
    "A4_C-5",
    "A5_F-16",
    "A6_TU-160",
    "A7_E-3",
    "A8_B-52",
    "A9_P-3C",
    "A10_B-1B",
    "A11_E-8",
    "A12_TU-22",
    "A13_F-15",
    "A14_KC-135",
    "A15_F-22",
    "A16_FA-18",
    "A17_TU-95",
    "A18_KC-10",
    "A19_SU-34",
    "A20_SU-24",
    "FSC",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class YoloAnnotation:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class ImageRecord:
    path: Path
    width: int
    height: int
    scene_id: str
    annotations: list[YoloAnnotation]


@dataclass
class SceneGroup:
    scene_id: str
    records: list[ImageRecord]
    instance_counts: list[int]
    image_counts: list[int]

    @property
    def num_images(self) -> int:
        return len(self.records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible FSOD-VFM baseline split from YOLO data."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/data/wangzijian/object_detection_datasets/datasets"),
        help="Dataset root containing images/train and labels/train.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/race"),
        help="Directory for split lists, COCO annotations, support JSON and statistics.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument(
        "--shots",
        nargs="+",
        default=["1", "3", "5", "10"],
        help="Support sizes to generate, for example: 1 3 5 10 all.",
    )
    return parser.parse_args()


def scene_id_from_stem(stem: str) -> str:
    """Recover a conservative source-scene identifier from an image stem."""
    if stem.startswith("MAR20_"):
        # No source-scene identifier is encoded in these names.
        return stem

    mission_match = re.search(r"(L[12]A\d+|L\d{8,})", stem)
    if mission_match:
        return mission_match.group(1)

    # FSC and any unknown cropped source stay grouped after removing crop index.
    return re.sub(r"_crop\d+$", "", stem, flags=re.IGNORECASE)


def parse_yolo_label(label_path: Path) -> list[YoloAnnotation]:
    annotations: list[YoloAnnotation] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"{label_path}:{line_number}: expected 5 fields, got {len(fields)}"
            )
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(
                f"{label_path}:{line_number}: invalid numeric value"
            ) from exc

        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(
                f"{label_path}:{line_number}: class id {class_id} is outside 0..24"
            )
        if not all(
            math.isfinite(value)
            for value in (x_center, y_center, width, height)
        ):
            raise ValueError(f"{label_path}:{line_number}: non-finite coordinate")
        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise ValueError(
                f"{label_path}:{line_number}: invalid normalized YOLO box"
            )
        annotations.append(
            YoloAnnotation(class_id, x_center, y_center, width, height)
        )
    if not annotations:
        raise ValueError(f"{label_path}: empty annotation file")
    return annotations


def load_records(dataset_root: Path) -> list[ImageRecord]:
    images_dir = dataset_root / "images" / "train"
    labels_dir = dataset_root / "labels" / "train"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {images_dir} and {labels_dir} to exist"
        )

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    label_paths = sorted(labels_dir.glob("*.txt"))
    image_by_stem = {path.stem: path for path in image_paths}
    label_by_stem = {path.stem: path for path in label_paths}
    missing_labels = sorted(image_by_stem.keys() - label_by_stem.keys())
    missing_images = sorted(label_by_stem.keys() - image_by_stem.keys())
    if missing_labels or missing_images:
        raise ValueError(
            "Image/label mismatch: "
            f"{len(missing_labels)} images without labels, "
            f"{len(missing_images)} labels without images"
        )

    records: list[ImageRecord] = []
    for stem in sorted(image_by_stem):
        image_path = image_by_stem[stem]
        with Image.open(image_path) as image:
            width, height = image.size
        records.append(
            ImageRecord(
                path=image_path,
                width=width,
                height=height,
                scene_id=scene_id_from_stem(stem),
                annotations=parse_yolo_label(label_by_stem[stem]),
            )
        )
    if not records:
        raise ValueError(f"No images found in {images_dir}")
    return records


def build_scene_groups(records: Iterable[ImageRecord]) -> list[SceneGroup]:
    grouped_records: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped_records[record.scene_id].append(record)

    groups: list[SceneGroup] = []
    for scene_id in sorted(grouped_records):
        scene_records = sorted(grouped_records[scene_id], key=lambda item: item.path.name)
        instance_counts = [0] * len(CLASS_NAMES)
        image_counts = [0] * len(CLASS_NAMES)
        for record in scene_records:
            present_classes = set()
            for annotation in record.annotations:
                instance_counts[annotation.class_id] += 1
                present_classes.add(annotation.class_id)
            for class_id in present_classes:
                image_counts[class_id] += 1
        groups.append(
            SceneGroup(scene_id, scene_records, instance_counts, image_counts)
        )
    return groups


def add_counts(left: list[int], right: list[int]) -> list[int]:
    return [a + b for a, b in zip(left, right)]


def subtract_counts(left: list[int], right: list[int]) -> list[int]:
    return [a - b for a, b in zip(left, right)]


def split_score(
    val_images: int,
    val_instances: list[int],
    val_class_images: list[int],
    target_images: float,
    target_instances: list[float],
    target_class_images: list[float],
) -> float:
    image_error = ((val_images - target_images) / max(target_images, 1.0)) ** 2
    instance_error = sum(
        ((actual - target) / max(target, 1.0)) ** 2
        for actual, target in zip(val_instances, target_instances)
    ) / len(CLASS_NAMES)
    class_image_error = sum(
        ((actual - target) / max(target, 1.0)) ** 2
        for actual, target in zip(val_class_images, target_class_images)
    ) / len(CLASS_NAMES)
    return 2.0 * image_error + instance_error + class_image_error


def create_grouped_split(
    groups: list[SceneGroup],
    val_ratio: float,
    seed: int,
    min_train_instances: int,
) -> tuple[list[SceneGroup], list[SceneGroup]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    total_images = sum(group.num_images for group in groups)
    total_instances = [
        sum(group.instance_counts[class_id] for group in groups)
        for class_id in range(len(CLASS_NAMES))
    ]
    total_class_images = [
        sum(group.image_counts[class_id] for group in groups)
        for class_id in range(len(CLASS_NAMES))
    ]
    if any(count <= min_train_instances for count in total_instances):
        raise ValueError(
            f"At least one class has no room for {min_train_instances}-shot support "
            "and a non-empty validation subset"
        )

    target_images = total_images * val_ratio
    target_instances = [count * val_ratio for count in total_instances]
    target_class_images = [count * val_ratio for count in total_class_images]
    # Build several seeded scene-level candidates and retain the one that best
    # matches both the requested image ratio and the per-class distributions.
    # This avoids a greedy bias toward very large source scenes.
    best_candidate = None
    for trial in range(512):
        trial_rng = random.Random(seed + trial * 1009)
        trial_selected = {
            group.scene_id for group in groups if trial_rng.random() < val_ratio
        }
        trial_groups = [
            group for group in groups if group.scene_id in trial_selected
        ]
        trial_images = sum(group.num_images for group in trial_groups)
        trial_instances = [
            sum(group.instance_counts[class_id] for group in trial_groups)
            for class_id in range(len(CLASS_NAMES))
        ]
        trial_class_images = [
            sum(group.image_counts[class_id] for group in trial_groups)
            for class_id in range(len(CLASS_NAMES))
        ]
        if any(count <= 0 for count in trial_instances):
            continue
        if any(
            total_instances[class_id] - trial_instances[class_id]
            < min_train_instances
            for class_id in range(len(CLASS_NAMES))
        ):
            continue
        trial_score = split_score(
            trial_images,
            trial_instances,
            trial_class_images,
            target_images,
            target_instances,
            target_class_images,
        )
        rank = (trial_score, trial)
        if best_candidate is None or rank < best_candidate[0]:
            best_candidate = (
                rank,
                trial_selected,
                trial_images,
                trial_instances,
                trial_class_images,
            )

    if best_candidate is None:
        raise RuntimeError("Unable to construct a valid grouped split candidate")

    _, selected, val_images, val_instances, val_class_images = best_candidate

    def can_add(group: SceneGroup) -> bool:
        return all(
            total_instances[class_id]
            - val_instances[class_id]
            - group.instance_counts[class_id]
            >= min_train_instances
            for class_id in range(len(CLASS_NAMES))
        )

    # Deterministic single-scene refinement first minimizes the distance to the
    # requested image ratio, then improves the class-distribution score.
    current_score = split_score(
        val_images,
        val_instances,
        val_class_images,
        target_images,
        target_instances,
        target_class_images,
    )
    for _ in range(50):
        best_move = None
        best_rank = (abs(val_images - target_images), current_score)
        for group in groups:
            if group.scene_id in selected:
                new_images = val_images - group.num_images
                new_instances = subtract_counts(val_instances, group.instance_counts)
                new_class_images = subtract_counts(
                    val_class_images, group.image_counts
                )
                if new_images <= 0 or any(count <= 0 for count in new_instances):
                    continue
            else:
                if not can_add(group):
                    continue
                new_images = val_images + group.num_images
                new_instances = add_counts(val_instances, group.instance_counts)
                new_class_images = add_counts(val_class_images, group.image_counts)
            candidate_score = split_score(
                new_images,
                new_instances,
                new_class_images,
                target_images,
                target_instances,
                target_class_images,
            )
            candidate_rank = (abs(new_images - target_images), candidate_score)
            if candidate_rank < best_rank:
                best_rank = candidate_rank
                best_move = (
                    group,
                    new_images,
                    new_instances,
                    new_class_images,
                )
        if best_move is None:
            break
        group, val_images, val_instances, val_class_images = best_move
        if group.scene_id in selected:
            selected.remove(group.scene_id)
        else:
            selected.add(group.scene_id)
        current_score = best_rank[1]

    val_groups = [group for group in groups if group.scene_id in selected]
    train_groups = [group for group in groups if group.scene_id not in selected]
    return train_groups, val_groups


def yolo_to_coco_bbox(
    annotation: YoloAnnotation, image_width: int, image_height: int
) -> list[float]:
    width = annotation.width * image_width
    height = annotation.height * image_height
    x_min = (annotation.x_center * image_width) - width / 2.0
    y_min = (annotation.y_center * image_height) - height / 2.0
    x_min = max(0.0, min(x_min, float(image_width)))
    y_min = max(0.0, min(y_min, float(image_height)))
    width = min(width, image_width - x_min)
    height = min(height, image_height - y_min)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("YOLO box became empty after conversion")
    return [round(x_min, 6), round(y_min, 6), round(width, 6), round(height, 6)]


def flatten_records(groups: Iterable[SceneGroup]) -> list[ImageRecord]:
    return sorted(
        (record for group in groups for record in group.records),
        key=lambda item: item.path.name,
    )


def make_coco_dataset(records: list[ImageRecord]) -> dict:
    images = []
    annotations = []
    annotation_id = 1
    for image_id, record in enumerate(records, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": record.path.name,
                "width": record.width,
                "height": record.height,
            }
        )
        for yolo_annotation in record.annotations:
            bbox = yolo_to_coco_bbox(
                yolo_annotation, record.width, record.height
            )
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": yolo_annotation.class_id + 1,
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 6),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "info": {
            "description": "Competition dataset prepared for FSOD-VFM",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": class_id + 1, "name": class_name}
            for class_id, class_name in enumerate(CLASS_NAMES)
        ],
    }


def make_support_sets(
    train_records: list[ImageRecord],
    dataset_root: Path,
    shots: list[str],
    seed: int,
) -> dict[str, dict[str, list[dict]]]:
    candidates: dict[int, list[dict]] = defaultdict(list)
    for record in train_records:
        relative_image_path = record.path.relative_to(dataset_root).as_posix()
        for annotation in record.annotations:
            candidates[annotation.class_id].append(
                {
                    "image": relative_image_path,
                    "bbox": yolo_to_coco_bbox(
                        annotation, record.width, record.height
                    ),
                }
            )

    ordered_candidates: dict[int, list[dict]] = {}
    for class_id in range(len(CLASS_NAMES)):
        class_candidates = sorted(
            candidates[class_id],
            key=lambda item: (item["image"], item["bbox"]),
        )
        random.Random(seed + class_id).shuffle(class_candidates)
        ordered_candidates[class_id] = class_candidates

    support_sets = {}
    for shot in shots:
        support_data = {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            sample_count = (
                len(ordered_candidates[class_id])
                if shot == "all"
                else int(shot)
            )
            if len(ordered_candidates[class_id]) < sample_count:
                raise ValueError(
                    f"Class {class_name} has only "
                    f"{len(ordered_candidates[class_id])} training instances, "
                    f"cannot create {sample_count}-shot support"
                )
            support_data[class_name] = ordered_candidates[class_id][:sample_count]
        support_sets[shot] = support_data
    return support_sets


def normalize_shots(raw_shots: list[str]) -> list[str]:
    shots = []
    for raw_shot in raw_shots:
        shot = str(raw_shot).lower()
        if shot != "all":
            try:
                numeric_shot = int(shot)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid shot value {raw_shot!r}; use a positive integer or 'all'"
                ) from exc
            if numeric_shot <= 0:
                raise ValueError("Shot values must be positive")
            shot = str(numeric_shot)
        if shot not in shots:
            shots.append(shot)
    return sorted(
        shots,
        key=lambda value: (value == "all", int(value) if value != "all" else 0),
    )


def count_records(records: Iterable[ImageRecord]) -> tuple[list[int], list[int]]:
    instance_counts = [0] * len(CLASS_NAMES)
    image_counts = [0] * len(CLASS_NAMES)
    for record in records:
        present_classes = set()
        for annotation in record.annotations:
            instance_counts[annotation.class_id] += 1
            present_classes.add(annotation.class_id)
        for class_id in present_classes:
            image_counts[class_id] += 1
    return instance_counts, image_counts


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    shots = normalize_shots(args.shots)
    numeric_shots = [int(shot) for shot in shots if shot != "all"]
    # "all" uses every instance remaining in train. It therefore does not
    # impose a fixed pre-split capacity. Keep the established 10-instance
    # constraint when all-shot is requested by itself for comparable splits.
    max_shot = max(numeric_shots, default=10)
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()

    records = load_records(dataset_root)
    groups = build_scene_groups(records)
    train_groups, val_groups = create_grouped_split(
        groups=groups,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_train_instances=max_shot,
    )
    train_records = flatten_records(train_groups)
    val_records = flatten_records(val_groups)

    train_scene_ids = {group.scene_id for group in train_groups}
    val_scene_ids = {group.scene_id for group in val_groups}
    if train_scene_ids & val_scene_ids:
        raise AssertionError("Scene leakage detected between train and validation")

    train_instances, train_class_images = count_records(train_records)
    val_instances, val_class_images = count_records(val_records)
    if any(count < max_shot for count in train_instances):
        raise AssertionError("Training split cannot supply the requested max shot")
    if any(count == 0 for count in val_instances):
        raise AssertionError("At least one class is missing from validation")

    support_sets = make_support_sets(
        train_records=train_records,
        dataset_root=dataset_root,
        shots=shots,
        seed=args.seed,
    )

    suffix = f"seed{args.seed}"
    write_lines(
        output_dir / "splits" / f"train_{suffix}.txt",
        (record.path.name for record in train_records),
    )
    write_lines(
        output_dir / "splits" / f"val_{suffix}.txt",
        (record.path.name for record in val_records),
    )
    write_json(
        output_dir / "annotations" / f"train_{suffix}.json",
        make_coco_dataset(train_records),
    )
    write_json(
        output_dir / "annotations" / f"val_{suffix}.json",
        make_coco_dataset(val_records),
    )
    for shot, support_data in support_sets.items():
        write_json(
            output_dir / "support" / f"{shot}shot_{suffix}.json",
            support_data,
        )

    statistics = {
        "dataset_root": str(dataset_root),
        "seed": args.seed,
        "val_ratio_requested": args.val_ratio,
        "val_ratio_actual": len(val_records) / len(records),
        "scene_grouping": {
            "total_groups": len(groups),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "scene_overlap": 0,
        },
        "splits": {
            "train": {
                "images": len(train_records),
                "instances": sum(train_instances),
            },
            "val": {
                "images": len(val_records),
                "instances": sum(val_instances),
            },
        },
        "classes": [
            {
                "yolo_id": class_id,
                "coco_id": class_id + 1,
                "name": class_name,
                "train_instances": train_instances[class_id],
                "val_instances": val_instances[class_id],
                "train_images": train_class_images[class_id],
                "val_images": val_class_images[class_id],
            }
            for class_id, class_name in enumerate(CLASS_NAMES)
        ],
        "support_sets": {
            f"{shot}shot": {
                class_name: len(samples)
                for class_name, samples in support_data.items()
            }
            for shot, support_data in support_sets.items()
        },
    }
    write_json(output_dir / f"split_statistics_{suffix}.json", statistics)

    print(f"Prepared dataset in: {output_dir}")
    print(
        f"Train: {len(train_records)} images, {sum(train_instances)} instances, "
        f"{len(train_groups)} scene groups"
    )
    print(
        f"Val:   {len(val_records)} images, {sum(val_instances)} instances, "
        f"{len(val_groups)} scene groups"
    )
    print(f"Actual validation ratio: {len(val_records) / len(records):.4f}")
    print("Per-class validation instances:")
    for class_name, count in zip(CLASS_NAMES, val_instances):
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()

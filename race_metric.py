"""Competition V1.5 Recall/FDR evaluation for COCO detection results."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from collections import defaultdict


SHIP_NAMES = {"HM", "LQS", "QHS", "MS"}
VEHICLE_NAMES = {"FSC"}
GROUP_NAMES = ("ship", "aircraft", "vehicle")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate detections with competition protocol V1.5."
    )
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--recall-requirement", type=float, default=0.85)
    parser.add_argument("--fdr-requirement", type=float, default=0.20)
    return parser.parse_args()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _target_group(category_name):
    if category_name in SHIP_NAMES:
        return "ship"
    if category_name in VEHICLE_NAMES:
        return "vehicle"
    return "aircraft"


def _bbox_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def _counts(tp, fp, ground_truth_count, score_threshold):
    fn = ground_truth_count - tp
    prediction_count = tp + fp
    recall = tp / ground_truth_count if ground_truth_count else 0.0
    fdr = fp / prediction_count if prediction_count else 0.0
    precision = tp / prediction_count if prediction_count else 0.0
    return {
        "score_threshold": score_threshold,
        "ground_truth_count": ground_truth_count,
        "prediction_count": prediction_count,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "recall": recall,
        "recall_percent": recall * 100.0,
        "FDR": fdr,
        "FDR_percent": fdr * 100.0,
        "precision": precision,
        "precision_percent": precision * 100.0,
    }


def _match_predictions(predictions, ground_truth_by_image, ground_truth_count, group=None):
    """Build cumulative operating points in descending score order."""
    matched = defaultdict(set)
    ordered_predictions = sorted(
        enumerate(predictions),
        key=lambda item: (-float(item[1]["score"]), item[0]),
    )
    tp = fp = 0
    points = []
    previous_score = None

    for _, prediction in ordered_predictions:
        if group is not None and prediction["_group"] != group:
            continue
        score = float(prediction["score"])
        if previous_score is not None and score != previous_score:
            points.append(_counts(tp, fp, ground_truth_count, previous_score))
        previous_score = score

        image_id = int(prediction["image_id"])
        best_index, best_iou = None, -1.0
        for index, ground_truth in enumerate(
            ground_truth_by_image.get(image_id, [])
        ):
            if index in matched[image_id]:
                continue
            if int(prediction["category_id"]) != int(ground_truth["category_id"]):
                continue
            if group is not None and ground_truth["_group"] != group:
                continue
            iou = _bbox_iou(prediction["bbox"], ground_truth["bbox"])
            iou_threshold = 0.35 if ground_truth["_group"] == "vehicle" else 0.50
            if iou >= iou_threshold and iou > best_iou:
                best_index, best_iou = index, iou

        if best_index is None:
            fp += 1
        else:
            tp += 1
            matched[image_id].add(best_index)

    if previous_score is not None:
        points.append(_counts(tp, fp, ground_truth_count, previous_score))
    return points, _counts(tp, fp, ground_truth_count, None)


def _metrics_at_threshold(points, ground_truth_count, threshold):
    selected = None
    for point in points:
        if point["score_threshold"] >= threshold:
            selected = point
        else:
            break
    if selected is None:
        return _counts(0, 0, ground_truth_count, threshold)
    result = dict(selected)
    result["score_threshold"] = threshold
    return result


def _recommendations(points, recall_requirement, fdr_requirement):
    under_fdr = [point for point in points if point["FDR"] <= fdr_requirement]
    above_recall = [
        point for point in points if point["recall"] >= recall_requirement
    ]
    passing = [
        point
        for point in points
        if point["recall"] >= recall_requirement
        and point["FDR"] <= fdr_requirement
    ]
    return {
        "best_recall_within_FDR_limit": (
            max(
                under_fdr,
                key=lambda point: (
                    point["recall"],
                    -point["FDR"],
                    point["score_threshold"],
                ),
            )
            if under_fdr
            else None
        ),
        "lowest_FDR_above_recall_limit": (
            min(
                above_recall,
                key=lambda point: (
                    point["FDR"],
                    -point["recall"],
                    -point["score_threshold"],
                ),
            )
            if above_recall
            else None
        ),
        "balanced_passing_point": (
            max(
                passing,
                key=lambda point: (
                    min(
                        point["recall"] - recall_requirement,
                        fdr_requirement - point["FDR"],
                    ),
                    point["recall"],
                    -point["FDR"],
                ),
            )
            if passing
            else None
        ),
        "has_passing_threshold": bool(passing),
    }


def _pass_status(metrics, recall_requirement, fdr_requirement):
    if metrics is None:
        return None
    recall_pass = metrics["recall"] >= recall_requirement
    fdr_pass = metrics["FDR"] <= fdr_requirement
    return {
        "recall_pass": recall_pass,
        "FDR_pass": fdr_pass,
        "accuracy_gate_pass": recall_pass and fdr_pass,
    }


def evaluate_race_predictions(
    ground_truth_data,
    predictions,
    score_threshold=None,
    recall_requirement=0.85,
    fdr_requirement=0.20,
):
    categories = {
        int(category["id"]): category["name"]
        for category in ground_truth_data["categories"]
    }
    ground_truth_by_image = defaultdict(list)
    group_counts = {group: 0 for group in GROUP_NAMES}
    for annotation in ground_truth_data["annotations"]:
        item = dict(annotation)
        item["_group"] = _target_group(categories[int(item["category_id"])])
        ground_truth_by_image[int(item["image_id"])].append(item)
        group_counts[item["_group"]] += 1

    prepared_predictions = []
    for prediction in predictions:
        category_id = int(prediction["category_id"])
        if category_id not in categories:
            raise ValueError(f"Unknown prediction category_id: {category_id}")
        if not math.isfinite(float(prediction["score"])):
            raise ValueError("Prediction score must be finite")
        item = dict(prediction)
        item["_group"] = _target_group(categories[category_id])
        prepared_predictions.append(item)

    total_ground_truth = len(ground_truth_data["annotations"])
    overall_points, overall_all = _match_predictions(
        prepared_predictions, ground_truth_by_image, total_ground_truth
    )
    overall_recommendations = _recommendations(
        overall_points, recall_requirement, fdr_requirement
    )
    threshold_source = "manual"
    if score_threshold is None:
        balanced_point = overall_recommendations["balanced_passing_point"]
        if balanced_point is not None:
            score_threshold = balanced_point["score_threshold"]
            threshold_source = "automatic_balanced_passing_point"
        else:
            threshold_source = "not_available"
    selected_overall = (
        _metrics_at_threshold(overall_points, total_ground_truth, score_threshold)
        if score_threshold is not None
        else None
    )
    result = {
        "protocol": {
            "name": "Competition scoring protocol V1.5",
            "overall_matching": (
                "all 25 subclasses are pooled, with exact category matching"
            ),
            "matching_order": "prediction score descending",
            "one_to_one_matching": True,
            "duplicate_predictions_are_FP": True,
            "IoU_thresholds": {
                "ship": 0.50,
                "aircraft": 0.50,
                "vehicle": 0.35,
            },
            "recall_requirement": recall_requirement,
            "FDR_requirement": fdr_requirement,
        },
        "all_predictions": {
            "overall": overall_all,
            "pass_status": _pass_status(
                overall_all, recall_requirement, fdr_requirement
            ),
        },
        "selected_threshold": {
            "threshold": score_threshold,
            "source": threshold_source,
            "overall": selected_overall,
            "pass_status": _pass_status(
                selected_overall, recall_requirement, fdr_requirement
            ),
        },
        "threshold_recommendations": overall_recommendations,
        "per_group": {},
    }

    for group in GROUP_NAMES:
        points, all_metrics = _match_predictions(
            prepared_predictions,
            ground_truth_by_image,
            group_counts[group],
            group=group,
        )
        selected = (
            _metrics_at_threshold(points, group_counts[group], score_threshold)
            if score_threshold is not None
            else None
        )
        result["per_group"][group] = {
            "all_predictions": all_metrics,
            "selected_threshold": selected,
            "threshold_recommendations": _recommendations(
                points, recall_requirement, fdr_requirement
            ),
        }
    return result


def _print_metrics(title, metrics):
    print(title)
    if metrics is None:
        print("  Not evaluated (no score threshold was provided).")
        return
    if metrics["score_threshold"] is not None:
        print(f"  Score threshold={metrics['score_threshold']:.9f}")
    print(
        f"  TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}, "
        f"predictions={metrics['prediction_count']}"
    )
    print(
        f"  Recall={metrics['recall_percent']:.2f}%, "
        f"FDR={metrics['FDR_percent']:.2f}%"
    )


def run_race_eval(
    gt_json_path,
    prediction_results,
    pred_json_path=None,
    score_threshold=None,
    output_dir="./results",
    recall_requirement=0.85,
    fdr_requirement=0.20,
):
    ground_truth_data = _load_json(gt_json_path)
    if isinstance(prediction_results, (str, os.PathLike)):
        predictions = _load_json(prediction_results)
        pred_json_path = str(prediction_results)
    else:
        predictions = prediction_results

    evaluation = evaluate_race_predictions(
        ground_truth_data,
        predictions,
        score_threshold,
        recall_requirement,
        fdr_requirement,
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "gt_json_path": gt_json_path,
        "pred_json_path": pred_json_path,
        **evaluation,
    }
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"race_eval_summary_{timestamp}.json"
    )
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n====== Evaluation ======")
    _print_metrics("All predictions:", evaluation["all_predictions"]["overall"])
    effective_threshold = evaluation["selected_threshold"]["threshold"]
    threshold_source = evaluation["selected_threshold"]["source"]
    _print_metrics(
        f"Selected threshold ({effective_threshold}, source={threshold_source}):",
        evaluation["selected_threshold"]["overall"],
    )
    recommendations = evaluation["threshold_recommendations"]
    _print_metrics(
        "Best recall while FDR <= limit:",
        recommendations["best_recall_within_FDR_limit"],
    )
    _print_metrics(
        "Lowest FDR while recall >= limit:",
        recommendations["lowest_FDR_above_recall_limit"],
    )
    print(
        "  A threshold satisfying both accuracy requirements exists: "
        f"{recommendations['has_passing_threshold']}"
    )
    selected_status = evaluation["selected_threshold"]["pass_status"]
    if selected_status is not None:
        print(
            "  Selected threshold accuracy gate pass: "
            f"{selected_status['accuracy_gate_pass']}"
        )
    print("\nPer-group metrics at the selected threshold:")
    for group in GROUP_NAMES:
        group_metrics = evaluation["per_group"][group]["selected_threshold"]
        if group_metrics is None:
            group_metrics = evaluation["per_group"][group]["all_predictions"]
        _print_metrics(f"  {group}:", group_metrics)
    print(f"Competition evaluation saved to: {output_path}")
    return summary


def main():
    args = parse_args()
    run_race_eval(
        args.gt_json,
        args.pred_json,
        pred_json_path=args.pred_json,
        score_threshold=args.score_threshold,
        output_dir=args.output_dir,
        recall_requirement=args.recall_requirement,
        fdr_requirement=args.fdr_requirement,
    )


if __name__ == "__main__":
    main()

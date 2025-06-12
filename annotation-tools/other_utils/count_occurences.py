import cv2
import json
import time
import os
import numpy as np
from sklearn.metrics import average_precision_score

# ---------------------- CONFIGURATION ----------------------
FOLDER_PATH = "RBK-kamper"
GAME_DIR = "RBK-Lillestrøm"
VIDEO_PATH = f"./{FOLDER_PATH}/{GAME_DIR}/game.mp4"
ANNOTATION_PATH = f"./{FOLDER_PATH}/{GAME_DIR}/labelsRBK.json"
PREDICTION_PATH = "./results_spotting.json"

LABEL_TO_ANALYZE = "SUCCESSFUL THROUGH BALL"
TOLERANCE_MS = 1000
CONFIDENCE_THRESHOLD = 0.1  # Used for qualitative TP/FP judgment
# -----------------------------------------------------------

def count_duplicate_fps(fp_list, tp_list, tolerance_ms=TOLERANCE_MS):
    duplicate_fp = 0
    tp_gt_positions = [gt["position"] for _, gt in tp_list]

    for fp in fp_list:
        for gt_pos in tp_gt_positions:
            if abs(fp["position"] - gt_pos) <= tolerance_ms:
                duplicate_fp += 1
                break

    return duplicate_fp

def load_annotations(json_path, target_label=LABEL_TO_ANALYZE):
    with open(json_path, "r") as f:
        data = json.load(f)
    annotations = []
    for ann in data.get("annotations", []):
        if ann["label"] != target_label:
            continue
        if ann.get("visibility", "visible") != "visible":
            continue
        annotations.append({
            "position": int(ann["position"]),
            "label": ann["label"],
            "team": ann.get("team", "unknown")
        })
    return annotations

def load_predictions(json_path, target_label=LABEL_TO_ANALYZE):
    with open(json_path, "r") as f:
        data = json.load(f)
    preds = []
    for pred in data.get("predictions", []):
        if pred["label"] != target_label:
            continue
        preds.append({
            "position": int(pred["position"]),
            "label": pred["label"],
            "team": pred.get("team", "unknown"),
            "confidence": pred.get("confidence", 0.0)
        })
    return preds

def classify_predictions(predictions, ground_truths, tolerance_ms=TOLERANCE_MS):
    matched_gt = set()
    matched_preds = set()
    tp, fp, fn = [], [], []

    for i, pred in enumerate(predictions):
        if pred["confidence"] < CONFIDENCE_THRESHOLD:
            continue
        for j, gt in enumerate(ground_truths):
            if abs(pred["position"] - gt["position"]) <= tolerance_ms and pred["team"] == gt["team"]:
                if j not in matched_gt and i not in matched_preds:
                    tp.append((pred, gt))
                    matched_gt.add(j)
                    matched_preds.add(i)
                    break

    for i, pred in enumerate(predictions):
        if i in matched_preds:
            continue
        if pred["confidence"] >= CONFIDENCE_THRESHOLD:
            fp.append(pred)

    for j, gt in enumerate(ground_truths):
        if j not in matched_gt:
            fn.append(gt)

    return tp, fp, fn

def compute_ap(predictions, ground_truths, tolerance_ms=TOLERANCE_MS):
    y_true = []
    y_scores = []

    matched_gt = set()
    for pred in sorted(predictions, key=lambda x: -x["confidence"]):
        match_found = False
        for j, gt in enumerate(ground_truths):
            if j in matched_gt:
                continue
            if abs(pred["position"] - gt["position"]) <= tolerance_ms and pred["team"] == gt["team"]:
                match_found = True
                matched_gt.add(j)
                break
        y_true.append(1 if match_found else 0)
        y_scores.append(pred["confidence"])

    if len(set(y_true)) < 2:
        return 0.0  # Avoid ill-defined AP

    return average_precision_score(y_true, y_scores)

def main():
    gts = load_annotations(ANNOTATION_PATH)
    preds = load_predictions(PREDICTION_PATH)
    tp, fp, fn = classify_predictions(preds, gts)

    sum_gt = len(gts)
    sum_preds = len(preds)
    sum_preds_conf = sum(1 for p in preds if p["confidence"] >= CONFIDENCE_THRESHOLD)
    num_tp = len(tp)
    num_fp = len(fp)
    num_fn = len(fn)
    duplicate_fp = count_duplicate_fps(fp, tp)

    precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
    recall = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0
    ap = compute_ap(preds, gts)

    print("Analysis for", LABEL_TO_ANALYZE)
    print("GT count:", sum_gt)
    print("Total predictions:", sum_preds)
    print("Predictions above confidence threshold:", sum_preds_conf)
    print("TP:", num_tp)
    print("FP:", num_fp, "where", duplicate_fp, "are duplicates for already-matched ground truth events:")
    print("FN:", num_fn)
    print("Precision:", f"{precision:.2f}")
    print("Recall:", f"{recall:.2f}")
    print("Average Precision (AP):", f"{ap:.2f}")

if __name__ == "__main__":
    main()

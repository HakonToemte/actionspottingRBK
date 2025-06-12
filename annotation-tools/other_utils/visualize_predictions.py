import cv2
import json
import time
import os

# ---------------------- CONFIGURATION ----------------------
FOLDER_PATH = "RBK-kamper"
GAME_DIR = "RBK-Lillestrøm"
VIDEO_PATH = f"./{FOLDER_PATH}/{GAME_DIR}/game.mp4"
ANNOTATION_PATH = f"./{FOLDER_PATH}/{GAME_DIR}/labelsRBK.json"
PREDICTION_PATH = "./results_spotting.json"

LABEL_TO_ANALYZE = "DEEP RUN"
TOLERANCE_MS = 3000
CONFIDENCE_THRESHOLD = 0.3
PRE_EVENT_BUFFER_MS = 5000  # Show 5 seconds before event
CLIP_DURATION_MS = 10000    # Total clip duration
START_FROM_MS = 0  # Set this to a timestamp in ms to skip clips before it

POSITIONS_TO_INCLUDE = [
    282400,
    1121840,
    4869200
]
# -----------------------------------------------------------

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

def load_predictions(json_path, target_label=LABEL_TO_ANALYZE, confidence_threshold=CONFIDENCE_THRESHOLD):
    with open(json_path, "r") as f:
        data = json.load(f)
    preds = []
    for pred in data.get("predictions", []):
        if pred["label"] != target_label:
            continue
        if pred.get("confidence", 0.0) < confidence_threshold:
            continue
        if POSITIONS_TO_INCLUDE and int(pred["position"]) not in POSITIONS_TO_INCLUDE:
            #pass
            continue
        preds.append({
            "position": int(pred["position"]),
            "label": pred["label"],
            "team": pred.get("team", "unknown"),
            "confidence": pred.get("confidence", 0.0)
        })
    return preds

def classify_predictions(predictions, ground_truths, tolerance_ms=TOLERANCE_MS):
    tp, fp, fn = [], [], []
    matched_gt = set()
    matched_preds = set()

    # Build a list of potential matches with distances
    potential_matches = []
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truths):
            distance = abs(pred["position"] - gt["position"])
            if distance <= tolerance_ms:
                potential_matches.append((distance, i, j))

    # Sort all potential matches by distance (closest first)
    potential_matches.sort()

    # Greedily assign closest unmatched prediction–GT pairs
    for distance, i, j in potential_matches:
        if i not in matched_preds and j not in matched_gt:
            tp.append((predictions[i], ground_truths[j]))
            matched_preds.add(i)
            matched_gt.add(j)

    # False positives: unmatched predictions above threshold
    for i, pred in enumerate(predictions):
        if i not in matched_preds:
            fp.append(pred)

    # False negatives: unmatched ground truth annotations
    for j, gt in enumerate(ground_truths):
        if j not in matched_gt:
            fn.append(gt)

    return tp, fp, fn


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    output_dir = "output_clips"
    os.makedirs(output_dir, exist_ok=True)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {VIDEO_PATH}")
        return
    writer = None
    output_path = "deep_run_combined.mp4"  # Final output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("set to 30")
        fps = 30.0
    print(fps, "fps")
    gts = load_annotations(ANNOTATION_PATH)
    preds = load_predictions(PREDICTION_PATH)
    tp, fp, fn = classify_predictions(preds, gts)

    clip_events = [(p, 'TP') for p, _ in tp] + [(p, 'FP') for p in fp]
    clip_events = [e for e in clip_events if e[0]['position'] >= START_FROM_MS]
    clip_events.sort(key=lambda x: (0 if x[1] == 'TP' else 1, x[0]['position']))
    clip_commentary = {
        282400: "False positive deep run detection triggered during a free kick",
        1121840: "True positive prediction of a deep run performed by the red team attacker",
        4869200: "False positive where the runs by the white team attackers are not purposeful enough to attack the space behind",
        # Add more entries as needed
    }
    

    for event, event_type in clip_events:
        start_time = max(0, event['position'] - PRE_EVENT_BUFFER_MS)
        end_time = start_time + CLIP_DURATION_MS
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if current_ms >= end_time:
                break

            progress = (current_ms - start_time) / (end_time - start_time)
            bar_width = frame.shape[1] - 20
            bar_x = 10
            bar_y = 10
            anchor_x = int(bar_x + (event['position'] - start_time) / (end_time - start_time) * bar_width)

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 12), (50, 50, 50), -1)

            # Progress fill
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(progress * bar_width), bar_y + 12), (200, 200, 200), -1)

            # Anchor line (red vertical line)
            cv2.line(frame, (anchor_x, bar_y - 2), (anchor_x, bar_y + 14), (0, 0, 255), 4)

            # Outer border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 12), (255, 255, 255), 1)

            label_text = f"{event['label']} ({event['team']}) [{event_type}]"
            if 'confidence' in event:
                label_text += f" ({event['confidence']:.2f})"

            color = (255, 255, 255) if event_type == 'TP' else (0, 0, 255)
            cv2.putText(frame, label_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            comment = clip_commentary.get(event['position'], "")
            if comment:
                y_offset = frame.shape[0] - 100
                cv2.putText(frame, comment, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            total_seconds = int(current_ms // 1000)
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            time_str = f"{minutes:02}:{seconds:02}"
            cv2.putText(frame, time_str, (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            writer.write(frame)
            #cv2.imshow("DEEP RUN Analysis", frame)

            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                while True:
                    if cv2.waitKey(0) & 0xFF == ord(' '):
                        break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import argparse
import json
import math
from pathlib import Path

import numpy as np


FEATURE_KEYS = ["bias", "conf", "size", "distance", "same_track", "alignment", "speed_consistency", "far_side"]
DEFAULT_WEIGHTS = {
    "bias": 0.0,
    "conf": 2.2,
    "size": 0.4,
    "distance": 2.4,
    "same_track": 0.9,
    "alignment": 0.8,
    "speed_consistency": 0.6,
    "far_side": 0.15,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small ball-candidate scoring model from tracker JSONL logs.")
    parser.add_argument("--jsonl", action="append", required=True, help="AI-debug JSONL log from track_ball_yolo.py. May be passed multiple times.")
    parser.add_argument("--output", required=True, help="Output model JSON path.")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--validation-split", type=float, default=0.2)
    return parser.parse_args()


def center_distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def selected_center(row):
    selector = (row.get("ball_debug") or {}).get("selector") or {}
    selected = selector.get("selected_candidate") or {}
    center = selected.get("center")
    if isinstance(center, list) and len(center) == 2:
        return center
    ball = row.get("ball") or {}
    center = ball.get("center")
    if isinstance(center, list) and len(center) == 2:
        return center
    return None


def rows_from_jsonl(path):
    samples = []
    groups = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            selector = (row.get("ball_debug") or {}).get("selector") or {}
            assist = selector.get("ai_assist") or {}
            candidates = assist.get("top_candidates") or []
            chosen_center = selected_center(row)
            if chosen_center is None or len(candidates) < 2:
                continue

            group_indexes = []
            for candidate in candidates:
                features = candidate.get("features") or {}
                if not all(key in features for key in FEATURE_KEYS):
                    continue
                center = candidate.get("center")
                if not isinstance(center, list) or len(center) != 2:
                    continue
                label = 1.0 if center_distance(center, chosen_center) <= 2.0 else 0.0
                x = [float(features[key]) for key in FEATURE_KEYS]
                group_indexes.append(len(samples))
                samples.append((x, label))
            if group_indexes and any(samples[index][1] == 1.0 for index in group_indexes):
                groups.append(group_indexes)
    return samples, groups


def split_groups(groups, validation_split):
    if not groups:
        return [], []
    validation_count = int(round(len(groups) * max(0.0, min(0.9, validation_split))))
    if validation_count <= 0:
        return groups, []
    return groups[:-validation_count], groups[-validation_count:]


def flatten_indexes(groups):
    indexes = []
    for group in groups:
        indexes.extend(group)
    return indexes


def sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def train(samples, train_indexes, epochs, learning_rate, l2):
    x = np.array([samples[index][0] for index in train_indexes], dtype=np.float64)
    y = np.array([samples[index][1] for index in train_indexes], dtype=np.float64)
    w = np.array([DEFAULT_WEIGHTS[key] for key in FEATURE_KEYS], dtype=np.float64)
    if len(x) == 0:
        return w
    for _ in range(max(1, epochs)):
        pred = sigmoid(x @ w)
        grad = (x.T @ (pred - y)) / len(x)
        grad += l2 * w
        grad[0] -= l2 * w[0]
        w -= learning_rate * grad
    return w


def evaluate(samples, groups, weights):
    if not groups:
        return {"groups": 0, "accuracy": None, "positive_rank1": 0, "mean_positive_rank": None}
    correct = 0
    positive_ranks = []
    for group in groups:
        scored = []
        for index in group:
            x, label = samples[index]
            score = float(np.dot(np.array(x), weights))
            scored.append((score, label))
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored[0][1] == 1.0:
            correct += 1
        for rank, (_score, label) in enumerate(scored, start=1):
            if label == 1.0:
                positive_ranks.append(rank)
                break
    return {
        "groups": len(groups),
        "accuracy": correct / len(groups),
        "positive_rank1": correct,
        "mean_positive_rank": sum(positive_ranks) / len(positive_ranks) if positive_ranks else None,
    }


def main():
    args = parse_args()
    all_samples = []
    all_groups = []
    for path in args.jsonl:
        samples, groups = rows_from_jsonl(path)
        offset = len(all_samples)
        all_samples.extend(samples)
        all_groups.extend([[index + offset for index in group] for group in groups])

    train_groups, validation_groups = split_groups(all_groups, args.validation_split)
    train_indexes = flatten_indexes(train_groups)
    weights = train(all_samples, train_indexes, args.epochs, args.learning_rate, args.l2)
    train_metrics = evaluate(all_samples, train_groups, weights)
    validation_metrics = evaluate(all_samples, validation_groups, weights)

    payload = {
        "model_type": "ball_candidate_logistic_v1",
        "feature_keys": FEATURE_KEYS,
        "weights": {key: float(weights[i]) for i, key in enumerate(FEATURE_KEYS)},
        "training": {
            "jsonl": args.jsonl,
            "samples": len(all_samples),
            "groups": len(all_groups),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(f"samples: {len(all_samples)}")
    print(f"groups: {len(all_groups)}")
    print(f"train: {train_metrics}")
    print(f"validation: {validation_metrics}")
    print(f"model: {output}")


if __name__ == "__main__":
    main()

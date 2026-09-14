"""Export image-space trajectory cards for candidate-free bounce events."""

import argparse
import csv
import json

from export_timeline_preroll_review import export_review


def read_events(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "near_existing_candidate", "status"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"review CSV missing fields: {sorted(missing)}")
        events = []
        for row in reader:
            if row["near_existing_candidate"] == "True":
                continue
            frame = int(row["frame"])
            events.append({
                "frame": frame,
                "kind": "candidate-free reversal",
                "note": (
                    f"direction change {row.get('direction_change_px_frame', '?')} px/frame; "
                    f"fit residual {row.get('fit_residual_px', '?')} px; "
                    f"evidence-review score {row.get('review_evidence_score', '?')}; "
                    "review only, not a bounce likelihood"
                ),
                "review_evidence_score": float(row.get("review_evidence_score") or 0),
                "trail_start_frame": max(1, frame - 60),
            })
    return sorted(events, key=lambda item: item["review_evidence_score"], reverse=True)


def load_diagnostics(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(event["frame"]): event for event in data.get("events", [])}


def load_labels(path):
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        labels = {}
        for row in rows:
            frame = (row.get("frame") or "").strip()
            if frame:
                scope = (row.get("review_scope") or "").strip()
                if scope and scope != "candidate_free_reversal":
                    raise ValueError(
                        f"labels CSV frame {frame} has incompatible review_scope {scope!r}"
                    )
                if frame in labels:
                    raise ValueError(f"labels CSV contains duplicate frame {frame}")
                labels[frame] = {
                    "label": row.get("label", ""),
                    "reviewer_confidence": row.get("reviewer_confidence", ""),
                    "note": row.get("note", ""),
                }
        return labels


def validate_label_frames(labels, events):
    """Reject labels that cannot be applied to this review population."""
    event_frames = {str(event["frame"]) for event in events}
    unknown = sorted(set(labels) - event_frames, key=int)
    if unknown:
        raise ValueError(
            "labels CSV contains frame(s) outside the candidate-free review population: "
            + ", ".join(unknown)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--diagnostics",
                        help="Optional review-only context JSON keyed by reversal frame.")
    parser.add_argument("--labels",
                        help="Optional previously exported review labels keyed by frame.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    events = read_events(args.reviews)
    diagnostics = load_diagnostics(args.diagnostics)
    labels = load_labels(args.labels)
    validate_label_frames(labels, events)
    for event in events:
        event.update(labels.get(str(event["frame"]), {}))
        context = diagnostics.get(str(event["frame"]))
        if not context:
            continue
        scene = context.get("scene_context") or {}
        players = scene.get("nearest_player") or {}
        player_distances = [details.get("distance_px") for details in players.values()
                            if details.get("distance_px") is not None]
        nearest_player = min(player_distances) if player_distances else None
        racket = scene.get("nearest_racket_distance_px")
        context_parts = ["context only; not a bounce label"]
        if nearest_player is not None:
            context_parts.append(f"nearest player {nearest_player:.1f}px")
        if racket is not None:
            context_parts.append(f"nearest racket {racket:.1f}px")
        event["note"] += "; " + ", ".join(context_parts)
    if not events:
        raise SystemExit("review CSV contains no candidate-free events")
    export_review([{
        "label": "tennis11 candidate-free reversals",
        "video": args.video,
        "jsonl": args.jsonl,
        "items": events,
    }], args.output)


if __name__ == "__main__":
    main()

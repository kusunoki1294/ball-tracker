import argparse
import atexit
import json
import math
import os
import statistics
from collections import deque

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


PLAYER_CLASS_ID = 0
BALL_COLOR = (0, 255, 255)
PLAYER_NEAR_COLOR = (255, 128, 0)
PLAYER_FAR_COLOR = (0, 200, 255)
GENERIC_COLOR = (0, 255, 0)
BOUNCE_COLOR = (0, 0, 255)
HIT_COLOR = (255, 0, 255)


class SimpleTracker:
    def __init__(self, max_distance):
        self.max_distance = max_distance
        self.next_track_id = 1
        self.tracks = {}

    def update(self, detections):
        updated_tracks = {}
        used_track_ids = set()

        for det in detections:
            best_track_id = None
            best_distance = None
            cx, cy = det["center"]

            for track_id, track in self.tracks.items():
                if track_id in used_track_ids:
                    continue
                if track["class_id"] != det["class_id"]:
                    continue

                tx, ty = track["center"]
                distance = math.hypot(cx - tx, cy - ty)
                if distance > self.max_distance:
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self.next_track_id
                self.next_track_id += 1

            det["track_id"] = best_track_id
            updated_tracks[best_track_id] = {
                "class_id": det["class_id"],
                "center": det["center"],
            }
            used_track_ids.add(best_track_id)

        self.tracks = updated_tracks
        return detections


class StationaryTrackFilter:
    def __init__(self, movement_px, static_frames):
        self.movement_px = movement_px
        self.static_frames = static_frames
        self.states = {}

    def filter(self, detections):
        active_ids = set()
        filtered = []

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                filtered.append(det)
                continue

            active_ids.add(track_id)
            center = det["center"]
            state = self.states.get(track_id)
            movement_threshold = self._movement_threshold(det)
            static_frame_limit = self._static_frame_limit(det)
            if state is None:
                self.states[track_id] = {"anchor_center": center, "last_center": center, "static_count": 0}
                filtered.append(det)
                continue

            anchor_center = state["anchor_center"]
            anchor_distance = math.hypot(center[0] - anchor_center[0], center[1] - anchor_center[1])
            if anchor_distance <= movement_threshold:
                static_count = state["static_count"] + 1
                next_state = {
                    "anchor_center": anchor_center,
                    "last_center": center,
                    "static_count": static_count,
                }
            else:
                static_count = 0
                next_state = {
                    "anchor_center": center,
                    "last_center": center,
                    "static_count": static_count,
                }
            self.states[track_id] = next_state
            if static_count < static_frame_limit:
                filtered.append(det)

        self.states = {track_id: self.states[track_id] for track_id in active_ids if track_id in self.states}
        return filtered

    def _movement_threshold(self, det):
        threshold = self.movement_px
        center_y = det["center"][1]
        bbox = det.get("bbox") or [0, 0, 0, 0]
        height = max(0, bbox[3] - bbox[1])
        if center_y < 420 or height <= 14:
            threshold *= 0.45
        elif center_y < 520 or height <= 22:
            threshold *= 0.7
        return max(1.5, threshold)

    def _static_frame_limit(self, det):
        limit = self.static_frames
        center_y = det["center"][1]
        bbox = det.get("bbox") or [0, 0, 0, 0]
        height = max(0, bbox[3] - bbox[1])
        if center_y < 420 or height <= 14:
            limit = int(round(limit * 1.6))
        elif center_y < 520 or height <= 22:
            limit = int(round(limit * 1.25))
        return max(2, limit)


class MovingBallFilter:
    def __init__(self, history_frames, min_travel_px):
        self.history_frames = history_frames
        self.min_travel_px = min_travel_px
        self.histories = {}
        self.last_debug = {}

    def filter(self, detections):
        active_ids = set()
        filtered = []
        debug = {
            "input_count": len(detections),
            "passed_bootstrap": 0,
            "passed_excursion": 0,
            "passed_seed_bypass": 0,
            "rejected_short_history": 0,
            "rejected_low_excursion": 0,
        }

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            active_ids.add(track_id)
            history = self.histories.setdefault(track_id, deque(maxlen=self.history_frames))
            history.append(tuple(det["center"]))

            if len(history) < 2:
                if self._bootstrap_detection(det):
                    filtered.append({**det, "motion_gate": "bootstrap"})
                    debug["passed_bootstrap"] += 1
                else:
                    debug["rejected_short_history"] += 1
                continue

            excursion = self._history_excursion(history)
            if excursion >= self._travel_threshold(det):
                filtered.append({**det, "motion_gate": "excursion"})
                debug["passed_excursion"] += 1
            elif self._seed_bypass_allowed(det, history, excursion):
                filtered.append({**det, "motion_gate": "seed_bypass"})
                debug["passed_seed_bypass"] += 1
            else:
                debug["rejected_low_excursion"] += 1

        self.histories = {track_id: self.histories[track_id] for track_id in active_ids if track_id in self.histories}
        self.last_debug = {
            **debug,
            "output_count": len(filtered),
        }
        return filtered

    def _bootstrap_detection(self, det):
        conf = det.get("conf") or 0.0
        x1, y1, x2, y2 = det["bbox"]
        height = y2 - y1
        return conf >= 0.55 and height <= 16

    def _travel_threshold(self, det):
        center_y = det["center"][1]
        threshold = self.min_travel_px
        bbox = det.get("bbox") or [0, 0, 0, 0]
        height = max(0, bbox[3] - bbox[1])
        if center_y < 420 or height <= 14:
            threshold *= 0.32
        elif center_y < 520 or height <= 22:
            threshold *= 0.55
        return max(2.0, threshold)

    def _history_excursion(self, history):
        if len(history) < 2:
            return 0.0
        max_distance = 0.0
        points = list(history)
        for i, (x0, y0) in enumerate(points):
            for x1, y1 in points[i + 1 :]:
                max_distance = max(max_distance, math.hypot(x1 - x0, y1 - y0))
        return max_distance

    def _seed_bypass_allowed(self, det, history, excursion):
        if len(history) < 2:
            return False
        center_y = det["center"][1]
        bbox = det.get("bbox") or [0, 0, 0, 0]
        height = max(0, bbox[3] - bbox[1])
        conf = det.get("conf") or 0.0
        if center_y >= 520 and height > 22:
            return False
        if conf < 0.20:
            return False
        if self._distinct_point_count(history) < 2:
            return False
        return excursion >= self._seed_bypass_threshold(det)

    def _seed_bypass_threshold(self, det):
        center_y = det["center"][1]
        bbox = det.get("bbox") or [0, 0, 0, 0]
        height = max(0, bbox[3] - bbox[1])
        threshold = self._travel_threshold(det)
        if center_y < 420 or height <= 14:
            return max(1.75, threshold * 0.45)
        return max(2.5, threshold * 0.55)

    def _distinct_point_count(self, history):
        distinct = set()
        for point in history:
            distinct.add((int(round(point[0])), int(round(point[1]))))
        return len(distinct)


class OnlineBallTrackAssist:
    FEATURE_KEYS = ["bias", "conf", "size", "distance", "same_track", "alignment", "speed_consistency", "far_side"]

    def __init__(self, learning_rate=0.02, influence=0.35, margin=0.15, update=True, model_path=None):
        self.learning_rate = max(0.0, float(learning_rate))
        self.influence = max(0.0, float(influence))
        self.margin = max(0.0, float(margin))
        self.update_enabled = bool(update)
        self.weights = {
            "bias": 0.0,
            "conf": 2.2,
            "size": 0.4,
            "distance": 2.4,
            "same_track": 0.9,
            "alignment": 0.8,
            "speed_consistency": 0.6,
            "far_side": 0.15,
        }
        self.model_path = None
        if model_path:
            self._load_model(model_path)
        self.last_debug = {}
        self.update_count = 0

    def rank(self, detections, predicted, last_ball, prev_ball, base_ranked):
        if not detections:
            self.last_debug = {"enabled": True, "candidate_count": 0}
            return []

        base_scores = {id(det): score for det, score in base_ranked}
        ranked = []
        entries = []
        for det in detections:
            features = self._features(det, predicted, last_ball, prev_ball)
            model_score = self._score_features(features)
            base_score = base_scores.get(id(det), 0.0)
            combined_score = base_score + (self.influence * model_score * 25.0)
            ranked.append((det, combined_score, model_score, features, base_score))
            entries.append(
                {
                    "track_id": det.get("track_id"),
                    "center": list(det["center"]),
                    "conf": round(det.get("conf") or 0.0, 3),
                    "base_score": round(base_score, 1),
                    "model_score": round(model_score, 3),
                    "combined_score": round(combined_score, 1),
                    "features": {key: round(value, 3) for key, value in features.items()},
                }
            )

        ranked.sort(key=lambda item: item[1], reverse=True)
        entries.sort(key=lambda item: item["combined_score"], reverse=True)
        self.last_debug = {
            "enabled": True,
            "candidate_count": len(detections),
            "influence": round(self.influence, 3),
            "updates": self.update_count,
            "model_path": self.model_path,
            "top_candidates": entries[:3],
        }
        return [(det, combined_score) for det, combined_score, _model_score, _features, _base_score in ranked]

    def learn(self, selected, detections, predicted, last_ball, prev_ball):
        if not self.update_enabled or self.learning_rate <= 0.0 or selected is None or len(detections) < 2:
            return

        selected_features = self._features(selected, predicted, last_ball, prev_ball)
        selected_score = self._score_features(selected_features)
        hardest_negative = None
        hardest_score = None
        hardest_features = None
        for det in detections:
            if det is selected:
                continue
            features = self._features(det, predicted, last_ball, prev_ball)
            score = self._score_features(features)
            if hardest_score is None or score > hardest_score:
                hardest_negative = det
                hardest_score = score
                hardest_features = features

        if hardest_negative is None or hardest_features is None:
            return
        if selected_score > hardest_score + self.margin:
            return

        for key in self.weights:
            self.weights[key] += self.learning_rate * (selected_features[key] - hardest_features[key])
        self.update_count += 1
        self.last_debug["last_update"] = {
            "selected_track_id": selected.get("track_id"),
            "negative_track_id": hardest_negative.get("track_id"),
            "selected_score": round(selected_score, 3),
            "negative_score": round(hardest_score, 3),
        }

    def _features(self, det, predicted, last_ball, prev_ball):
        center = tuple(det["center"])
        conf = float(det.get("conf") or 0.0)
        bbox = det.get("bbox") or [0, 0, 0, 0]
        height = max(0.0, float(bbox[3] - bbox[1]))
        distance = math.hypot(center[0] - predicted[0], center[1] - predicted[1]) if predicted is not None else 0.0
        same_track = 0.0
        if last_ball is not None and det.get("track_id") is not None and det.get("track_id") == last_ball.get("track_id"):
            same_track = 1.0

        alignment = 0.0
        speed_consistency = 0.0
        if last_ball is not None and prev_ball is not None:
            last_center = tuple(last_ball["center"])
            prev_center = tuple(prev_ball["center"])
            pred_vx = last_center[0] - prev_center[0]
            pred_vy = last_center[1] - prev_center[1]
            cand_vx = center[0] - last_center[0]
            cand_vy = center[1] - last_center[1]
            pred_speed = math.hypot(pred_vx, pred_vy)
            cand_speed = math.hypot(cand_vx, cand_vy)
            if pred_speed > 1.0 and cand_speed > 1.0:
                alignment = ((pred_vx * cand_vx) + (pred_vy * cand_vy)) / (pred_speed * cand_speed)
                speed_consistency = -min(2.0, abs(cand_speed - pred_speed) / max(pred_speed, 10.0))

        return {
            "bias": 1.0,
            "conf": min(1.0, max(0.0, conf)),
            "size": min(1.0, height / 18.0),
            "distance": -min(2.0, distance / 220.0),
            "same_track": same_track,
            "alignment": max(-1.0, min(1.0, alignment)),
            "speed_consistency": speed_consistency,
            "far_side": 1.0 if center[1] < 420 else 0.0,
        }

    def _load_model(self, model_path):
        with open(model_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        weights = payload.get("weights") if isinstance(payload, dict) else None
        if not isinstance(weights, dict):
            raise ValueError(f"Invalid ball AI model file: {model_path}")
        for key in self.FEATURE_KEYS:
            if key in weights:
                self.weights[key] = float(weights[key])
        self.model_path = str(model_path)

    def _score_features(self, features):
        return sum(self.weights[key] * features[key] for key in self.FEATURE_KEYS)


class BallSelector:
    def __init__(
        self,
        seed_confirm_frames=2,
        seed_min_travel_px=4.0,
        seed_match_distance=90.0,
        pending_miss_tolerance=1,
        coast_frames=2,
        max_jump_px=280.0,
        max_prediction_error_px=240.0,
        max_side_flip_jump_px=380.0,
        ai_assist=None,
    ):
        self.last_ball = None
        self.prev_ball = None
        self.missed_frames = 0
        self.seed_confirm_frames = max(1, int(seed_confirm_frames))
        self.seed_min_travel_px = max(0.0, float(seed_min_travel_px))
        self.seed_match_distance = max(1.0, float(seed_match_distance))
        self.pending_miss_tolerance = max(0, int(pending_miss_tolerance))
        self.coast_frames = max(0, int(coast_frames))
        self.pending_ball = None
        self.pending_seen_frames = 0
        self.pending_missed_frames = 0
        self.pending_history = deque(maxlen=max(2, self.seed_confirm_frames + self.pending_miss_tolerance + 1))
        self.max_jump_px = max(1.0, float(max_jump_px))
        self.max_prediction_error_px = max(1.0, float(max_prediction_error_px))
        self.max_side_flip_jump_px = max(1.0, float(max_side_flip_jump_px))
        self.ai_assist = ai_assist
        self.last_debug = {}

    def select(self, ball_detections):
        self.last_debug = {
            "mode": "track" if self.last_ball is not None else "seed",
            "candidate_count": len(ball_detections),
            "missed_frames_before": self.missed_frames,
            "active_track_id": self.last_ball.get("track_id") if self.last_ball is not None else None,
        }

        if not ball_detections:
            self.missed_frames += 1
            coasted_ball = None
            if self.last_ball is not None and self.missed_frames <= self.coast_frames:
                coasted_ball = self._coasted_ball()
            if self.missed_frames > 2:
                self.prev_ball = self.last_ball
                self.last_ball = None
            reason = "no_candidates"
            if self.last_ball is None and self.pending_ball is not None:
                self.pending_missed_frames += 1
                reason = "seed_pending_miss"
                if self.pending_missed_frames > self.pending_miss_tolerance:
                    self._clear_pending()
                    reason = "seed_pending_expired"
            self.last_debug.update(
                {
                    "decision": "no_ball",
                    "reason": reason,
                    "missed_frames_after": self.missed_frames,
                    "coasted": self._ball_debug(coasted_ball),
                    "pending": self._pending_debug(),
                }
            )
            if coasted_ball is not None:
                self.last_debug.update({"decision": "selected", "reason": "coasted"})
                return coasted_ball
            return None

        if self.last_ball is None:
            ranked = sorted(
                ((det, self._initial_score(det)) for det in ball_detections),
                key=lambda item: item[1],
                reverse=True,
            )
            chosen, chosen_score = ranked[0]
            matched_pending = self._pending_matches(chosen)
            if matched_pending:
                self.pending_seen_frames += 1
            else:
                self.pending_seen_frames = 1
                self.pending_missed_frames = 0
                self.pending_history.clear()
            self.pending_ball = chosen
            self.pending_missed_frames = 0
            self.pending_history.append(tuple(chosen["center"]))
            pending_travel = self._pending_travel()
            pending_threshold = self._seed_travel_threshold(chosen)
            pending_confirm_frames = self._seed_confirm_frames_required(chosen)

            self.last_debug.update(
                {
                    "top_candidates": self._candidate_debug(ranked),
                    "selected_candidate": self._candidate_entry(chosen, chosen_score),
                    "pending": self._pending_debug(
                        travel=pending_travel,
                        travel_threshold=pending_threshold,
                        confirm_frames=pending_confirm_frames,
                    ),
                }
            )

            if self.pending_seen_frames < pending_confirm_frames:
                self.last_debug.update({"decision": "pending", "reason": "seed_wait_confirm"})
                return None
            if pending_travel < pending_threshold:
                self.last_debug.update({"decision": "pending", "reason": "seed_wait_travel"})
                return None

            self._record(chosen)
            self._clear_pending()
            self.last_debug.update({"decision": "selected", "reason": "seed_confirmed"})
            return chosen

        predicted = self._predicted_center()
        ranked = sorted(
            ((det, self._tracking_score(det, predicted)) for det in ball_detections),
            key=lambda item: item[1],
            reverse=True,
        )
        ai_used = False
        if self.ai_assist is not None:
            ai_ranked = self.ai_assist.rank(ball_detections, predicted, self.last_ball, self.prev_ball, ranked)
            if ai_ranked:
                ranked = ai_ranked
                ai_used = True
        chosen = None
        chosen_score = None
        rejected_candidates = []
        for candidate, candidate_score in ranked:
            rejection = self._physical_jump_rejection(candidate, predicted)
            if rejection is None:
                chosen = candidate
                chosen_score = candidate_score
                break
            rejected_candidates.append(
                {
                    "candidate": self._candidate_entry(candidate, candidate_score),
                    "rejection": rejection,
                }
            )
        if chosen is None:
            self.missed_frames += 1
            self.last_debug.update(
                {
                    "predicted_center": [int(round(predicted[0])), int(round(predicted[1]))],
                    "top_candidates": self._candidate_debug(ranked),
                    "rejected_candidates": rejected_candidates[:3],
                    "decision": "no_ball",
                    "reason": "physical_jump_rejected",
                    "missed_frames_after": self.missed_frames,
                    "ai_assist": self.ai_assist.last_debug if self.ai_assist is not None else None,
                }
            )
            return None
        self.last_debug.update(
            {
                "predicted_center": [int(round(predicted[0])), int(round(predicted[1]))],
                "top_candidates": self._candidate_debug(ranked),
                "selected_candidate": self._candidate_entry(chosen, chosen_score),
                "rejected_candidates": rejected_candidates[:3],
                "ai_assist": self.ai_assist.last_debug if self.ai_assist is not None else None,
            }
        )
        previous_last = self.last_ball
        previous_prev = self.prev_ball
        self._clear_pending()
        self._record(chosen)
        if self.ai_assist is not None:
            self.ai_assist.learn(chosen, ball_detections, predicted, previous_last, previous_prev)
            self.last_debug["ai_assist"] = self.ai_assist.last_debug
        self.last_debug.update({"decision": "selected", "reason": "tracked_ai" if ai_used else "tracked"})
        return chosen

    def _record(self, ball):
        self.prev_ball = self.last_ball
        self.last_ball = ball
        self.missed_frames = 0

    def _predicted_center(self):
        last_center = tuple(self.last_ball["center"])
        if self.prev_ball is None:
            return last_center
        prev_center = tuple(self.prev_ball["center"])
        vx = last_center[0] - prev_center[0]
        vy = last_center[1] - prev_center[1]
        return (last_center[0] + vx, last_center[1] + vy)

    def _coasted_ball(self):
        predicted = self._predicted_center()
        last_bbox = self.last_ball.get("bbox") or [0, 0, 0, 0]
        width = max(4, int(last_bbox[2] - last_bbox[0]))
        height = max(4, int(last_bbox[3] - last_bbox[1]))
        cx = int(round(predicted[0]))
        cy = int(round(predicted[1]))
        half_w = width // 2
        half_h = height // 2
        return {
            **self.last_ball,
            "conf": min(float(self.last_ball.get("conf") or 0.0), 0.05),
            "bbox": [cx - half_w, cy - half_h, cx - half_w + width, cy - half_h + height],
            "center": [cx, cy],
            "interpolated": True,
            "motion_gate": "coast",
        }

    def _initial_score(self, det):
        height = det["bbox"][3] - det["bbox"][1]
        conf = det.get("conf") or 0.0
        score = conf * 150.0 + min(height, 16) * 1.5
        if det["center"][1] < 420:
            score += 12.0
        return score

    def _tracking_score(self, det, predicted):
        center = tuple(det["center"])
        conf = det.get("conf") or 0.0
        height = det["bbox"][3] - det["bbox"][1]
        distance = math.hypot(center[0] - predicted[0], center[1] - predicted[1])
        pred_far_side = predicted[1] < 420 and center[1] < 420

        score = conf * 140.0 + min(height, 16) * 1.5
        distance_penalty = 0.95
        if pred_far_side:
            distance_penalty = 0.42
        score -= distance * distance_penalty

        if pred_far_side and distance > 42.0:
            score -= (distance - 42.0) * 1.35
        if pred_far_side and conf < 0.4 and distance > 58.0:
            score -= 55.0
        if pred_far_side and conf < 0.25 and distance > 46.0:
            score -= 90.0

        if det.get("track_id") is not None and det.get("track_id") == self.last_ball.get("track_id"):
            score += 32.0
            if distance < 90.0:
                score += 14.0

        if self.prev_ball is not None:
            last_center = tuple(self.last_ball["center"])
            prev_center = tuple(self.prev_ball["center"])
            pred_vx = last_center[0] - prev_center[0]
            pred_vy = last_center[1] - prev_center[1]
            cand_vx = center[0] - last_center[0]
            cand_vy = center[1] - last_center[1]
            pred_speed = math.hypot(pred_vx, pred_vy)
            cand_speed = math.hypot(cand_vx, cand_vy)
            if pred_speed > 1.0 and cand_speed > 1.0:
                alignment = ((pred_vx * cand_vx) + (pred_vy * cand_vy)) / (pred_speed * cand_speed)
                score += alignment * (24.0 if center[1] < 420 else 16.0)
                if alignment < -0.35:
                    score -= 18.0

        if center[1] < 420:
            score += 10.0

        return score

    def _physical_jump_rejection(self, det, predicted):
        if self.last_ball is None:
            return None
        center = tuple(det["center"])
        conf = det.get("conf") or 0.0
        last_center = tuple(self.last_ball["center"])
        last_distance = math.hypot(center[0] - last_center[0], center[1] - last_center[1])
        prediction_error = math.hypot(center[0] - predicted[0], center[1] - predicted[1])
        missing_allowance = max(0, self.missed_frames)
        max_jump = self.max_jump_px + (missing_allowance * 95.0)
        max_prediction_error = self.max_prediction_error_px + (missing_allowance * 105.0)
        max_side_flip_jump = self.max_side_flip_jump_px + (missing_allowance * 85.0)
        last_side = "far" if last_center[1] < 420 else "near"
        center_side = "far" if center[1] < 420 else "near"
        side_flip = last_side != center_side

        if last_distance > max_jump:
            return {
                "reason": "max_frame_jump",
                "last_distance_px": round(last_distance, 1),
                "limit_px": round(max_jump, 1),
                "missing_frames": missing_allowance,
            }
        if prediction_error > max_prediction_error:
            return {
                "reason": "prediction_error",
                "prediction_error_px": round(prediction_error, 1),
                "limit_px": round(max_prediction_error, 1),
                "missing_frames": missing_allowance,
            }
        if side_flip and last_distance > max_side_flip_jump:
            return {
                "reason": "side_flip_jump",
                "last_distance_px": round(last_distance, 1),
                "limit_px": round(max_side_flip_jump, 1),
                "from_side": last_side,
                "to_side": center_side,
                "missing_frames": missing_allowance,
            }
        if last_center[1] < 420 and center[1] < 420:
            distance = prediction_error
            if conf < 0.25 and distance > 46.0:
                return {
                    "reason": "low_conf_far_prediction_error",
                    "prediction_error_px": round(distance, 1),
                    "limit_px": 46.0,
                    "conf": round(conf, 3),
                }
            if conf < 0.35 and distance > 62.0:
                return {
                    "reason": "medium_conf_far_prediction_error",
                    "prediction_error_px": round(distance, 1),
                    "limit_px": 62.0,
                    "conf": round(conf, 3),
                }
        return None

    def _clear_pending(self):
        self.pending_ball = None
        self.pending_seen_frames = 0
        self.pending_missed_frames = 0
        self.pending_history.clear()

    def _pending_matches(self, det):
        if self.pending_ball is None:
            return False
        pending_track_id = self.pending_ball.get("track_id")
        track_id = det.get("track_id")
        if pending_track_id is not None and track_id is not None and pending_track_id == track_id:
            return True
        pending_center = tuple(self.pending_ball["center"])
        center = tuple(det["center"])
        return math.hypot(center[0] - pending_center[0], center[1] - pending_center[1]) <= self.seed_match_distance

    def _pending_travel(self):
        if len(self.pending_history) < 2:
            return 0.0
        start_x, start_y = self.pending_history[0]
        end_x, end_y = self.pending_history[-1]
        return math.hypot(end_x - start_x, end_y - start_y)

    def _pending_debug(self, travel=None, travel_threshold=None, confirm_frames=None):
        if self.pending_ball is None:
            return None
        return {
            "track_id": self.pending_ball.get("track_id"),
            "center": list(self.pending_ball["center"]),
            "seen_frames": self.pending_seen_frames,
            "missed_frames": self.pending_missed_frames,
            "travel_px": round(self._pending_travel() if travel is None else travel, 1),
            "travel_threshold_px": round(
                self._seed_travel_threshold(self.pending_ball) if travel_threshold is None else travel_threshold,
                1,
            ),
            "confirm_frames_required": (
                self._seed_confirm_frames_required(self.pending_ball) if confirm_frames is None else confirm_frames
            ),
        }

    def _candidate_entry(self, det, score):
        return {
            "track_id": det.get("track_id"),
            "center": list(det["center"]),
            "conf": round(det.get("conf") or 0.0, 3),
            "score": round(score, 1),
        }

    def _ball_debug(self, det):
        if det is None:
            return None
        return {
            "track_id": det.get("track_id"),
            "center": list(det["center"]),
            "conf": round(det.get("conf") or 0.0, 3),
            "interpolated": bool(det.get("interpolated")),
        }

    def _candidate_debug(self, ranked, limit=3):
        return [self._candidate_entry(det, score) for det, score in ranked[:limit]]

    def _seed_travel_threshold(self, det):
        threshold = self.seed_min_travel_px
        center_y = det["center"][1]
        bbox = det.get("bbox") or [0, 0, 0, 0]
        height = max(0, bbox[3] - bbox[1])
        if center_y < 420 or height <= 14:
            threshold *= 0.45
        elif center_y < 520 or height <= 22:
            threshold *= 0.65
        if det.get("motion_gate") == "seed_bypass":
            threshold = max(threshold * 1.75, 3.0)
        return max(1.75, threshold)

    def _seed_confirm_frames_required(self, det):
        if det.get("motion_gate") == "seed_bypass":
            return max(self.seed_confirm_frames + 1, 3)
        return self.seed_confirm_frames


def point_to_bbox_distance(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return math.hypot(dx, dy)


def filter_ball_candidates_by_contact_zone(
    detections,
    near_player,
    racket_detections,
    player_hit_margin_px,
    racket_hit_margin_px,
):
    if not detections:
        return detections

    filtered = []
    near_strike_zone = None
    if near_player is not None:
        x1, y1, x2, y2 = near_player["bbox"]
        height = y2 - y1
        near_strike_zone = [
            x1 - max(6.0, player_hit_margin_px * 0.75),
            y1 - max(38.0, height * 0.12),
            x2 + max(24.0, player_hit_margin_px * 3.0),
            y1 + (height * 0.35),
        ]

    racket_margin = max(10.0, racket_hit_margin_px * 0.75)
    for det in detections:
        point = ball_contact_point(det) or det.get("center")
        if point is None:
            filtered.append(det)
            continue

        if near_strike_zone is not None and point_to_bbox_distance(point, near_strike_zone) <= 0:
            continue
        if any(point_to_bbox_distance(point, racket["bbox"]) <= racket_margin for racket in racket_detections):
            continue
        filtered.append(det)
    return filtered


def point_to_segment_distance(point, segment_start, segment_end):
    px, py = point
    ax, ay = segment_start
    bx, by = segment_end
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= 0:
        return math.hypot(px - ax, py - ay)
    u = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    qx = ax + (u * abx)
    qy = ay + (u * aby)
    return math.hypot(px - qx, py - qy)


def average_point(points):
    valid = [point for point in points if point is not None]
    if not valid:
        return None
    x = sum(point[0] for point in valid) / len(valid)
    y = sum(point[1] for point in valid) / len(valid)
    return (x, y)


def ball_contact_point(ball):
    if not ball:
        return None
    bbox = ball.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, _y1, x2, y2 = bbox
        return (float(x1 + x2) / 2.0, float(y2))
    center = ball.get("center")
    if isinstance(center, list) and len(center) == 2:
        return (float(center[0]), float(center[1]))
    return None


def order_court_corners(points):
    points = sorted(points, key=lambda p: p[1])
    far = points[:2]
    near = points[2:]
    far_left, far_right = sorted(far, key=lambda p: p[0])
    near_left, near_right = sorted(near, key=lambda p: p[0])
    return [near_left, near_right, far_right, far_left]


def get_mini_court_layout(frame_shape, size, margin):
    frame_h, frame_w = frame_shape[:2]
    court_len = 78.0
    court_wid = 36.0
    singles_wid = 27.0
    aspect = court_len / court_wid
    overlay_w = max(120, int(size))
    overlay_h = int(overlay_w * aspect)

    if overlay_h + margin * 2 > frame_h:
        overlay_h = max(160, frame_h - margin * 2)
        overlay_w = int(overlay_h / aspect)
    if overlay_w + margin * 2 > frame_w:
        overlay_w = max(120, frame_w - margin * 2)
        overlay_h = int(overlay_w * aspect)

    x1 = max(0, frame_w - overlay_w - margin)
    y1 = margin
    x2 = min(frame_w - 1, x1 + overlay_w)
    y2 = min(frame_h - 1, y1 + overlay_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "overlay_w": x2 - x1,
        "overlay_h": y2 - y1,
        "court_len": court_len,
        "court_wid": court_wid,
        "singles_wid": singles_wid,
    }


def mini_court_point(layout, xw, yw):
    px = int(layout["x1"] + (xw / layout["court_wid"]) * layout["overlay_w"])
    py = int(layout["y1"] + (yw / layout["court_len"]) * layout["overlay_h"])
    return (px, py)


def draw_mini_court(frame, enabled, size, margin):
    if not enabled:
        return None

    layout = get_mini_court_layout(frame.shape, size=size, margin=margin)
    if layout is None:
        return None

    x1 = layout["x1"]
    y1 = layout["y1"]
    x2 = layout["x2"]
    y2 = layout["y2"]
    overlay_w = layout["overlay_w"]
    overlay_h = layout["overlay_h"]
    panel = frame[y1:y2, x1:x2]
    if panel.size == 0:
        return None

    tint = panel.copy()
    tint[:] = (24, 42, 18)
    cv2.addWeighted(tint, 0.72, panel, 0.28, 0, panel)

    color = (235, 245, 235)
    thick = 1 if overlay_w < 180 else 2

    def pt(xw, yl):
        return mini_court_point(layout, xw, yl)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 220, 200), 1)

    court_len = layout["court_len"]
    court_wid = layout["court_wid"]
    singles_wid = layout["singles_wid"]
    singles_left = (court_wid - singles_wid) / 2.0
    singles_right = singles_left + singles_wid
    service_y_top = 21.0
    service_y_bottom = court_len - 21.0
    net_y = court_len / 2.0
    center_x = court_wid / 2.0

    cv2.rectangle(frame, pt(0.0, 0.0), pt(court_wid, court_len), color, thick)
    cv2.rectangle(frame, pt(singles_left, 0), pt(singles_right, court_len), color, thick)
    cv2.line(frame, pt(singles_left, service_y_top), pt(singles_right, service_y_top), color, thick)
    cv2.line(frame, pt(singles_left, service_y_bottom), pt(singles_right, service_y_bottom), color, thick)
    cv2.line(frame, pt(singles_left, net_y), pt(singles_right, net_y), color, thick)
    cv2.line(frame, pt(center_x, service_y_top), pt(center_x, service_y_bottom), color, thick)

    mark_len = 2.0
    cv2.line(frame, pt(center_x, 0), pt(center_x, mark_len), color, thick)
    cv2.line(frame, pt(center_x, court_len - mark_len), pt(center_x, court_len), color, thick)
    return layout


def draw_mini_court_points(frame, layout, ball_world_point, bounce_world_points):
    if layout is None:
        return

    def in_bounds(world_point):
        if world_point is None:
            return False
        xw, yw = world_point
        return 0.0 <= xw <= layout["court_wid"] and 0.0 <= yw <= layout["court_len"]

    for world_point in bounce_world_points[-8:]:
        if not in_bounds(world_point):
            continue
        point = mini_court_point(layout, world_point[0], world_point[1])
        cv2.circle(frame, point, 4, BOUNCE_COLOR, -1)
        cv2.circle(frame, point, 7, (255, 255, 255), 1)

    if in_bounds(ball_world_point):
        point = mini_court_point(layout, ball_world_point[0], ball_world_point[1])
        cv2.circle(frame, point, 6, BALL_COLOR, -1)
        cv2.circle(frame, point, 9, (40, 40, 40), 1)


def load_court_calibration(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    points = data.get("points")
    net_points = data.get("net_points")
    if not isinstance(points, list) or len(points) != 4:
        return None

    try:
        court_points = order_court_corners([(float(x), float(y)) for x, y in points])
        net = None
        if isinstance(net_points, list) and len(net_points) == 2:
            net = [(float(x), float(y)) for x, y in net_points]
    except Exception:
        return None
    return {"points": court_points, "net_points": net}


def build_inverse_court_homography(court_calib):
    if not court_calib or court_calib.get("points") is None:
        return None
    # Calibration points are stored as near-left, near-right, far-right, far-left.
    # Map them to world coordinates with the far baseline at y=0 and the near
    # baseline at y=78 so near/far classification and the mini-court share the
    # same orientation.
    world = np.array([[0.0, 78.0], [36.0, 78.0], [36.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    image = np.array(court_calib["points"], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(world, image)
    try:
        return np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None


def calibration_fits_frame(court_calib, frame_shape, margin_px=24):
    if not court_calib:
        return False
    frame_h, frame_w = frame_shape[:2]
    all_points = list(court_calib.get("points") or [])
    all_points.extend(court_calib.get("net_points") or [])
    if not all_points:
        return False
    for x, y in all_points:
        if x < -margin_px or x > frame_w + margin_px or y < -margin_px or y > frame_h + margin_px:
            return False
    return True


def project_to_court_world(center, inv_homography):
    if center is None or inv_homography is None:
        return None
    point = np.array([[center]], dtype=np.float32)
    world = cv2.perspectiveTransform(point, inv_homography)[0][0]
    return float(world[0]), float(world[1])


def world_point_in_court(world_point, margin=0.0):
    if world_point is None:
        return False
    xw, yw = world_point
    return -margin <= xw <= 36.0 + margin and -margin <= yw <= 78.0 + margin


def ball_world_point_in_tracking_bounds(world_point):
    if world_point is None:
        return True
    xw, yw = world_point
    if yw < -60.0:
        return 22.0 <= xw <= 32.5 and -105.0 <= yw <= -60.0
    if yw < 0.0:
        return 4.0 <= xw <= 32.0 and -60.0 <= yw <= 96.0
    return -8.0 <= xw <= 44.0 and -24.0 <= yw <= 96.0


def filter_ball_candidates_by_court(detections, inv_homography):
    if inv_homography is None or not detections:
        return detections

    filtered = []
    for det in detections:
        contact_point = ball_contact_point(det)
        center_point = det.get("center")
        contact_world = project_to_court_world(contact_point, inv_homography)
        center_world = project_to_court_world(center_point, inv_homography)
        if ball_world_point_in_tracking_bounds(contact_world) or ball_world_point_in_tracking_bounds(center_world):
            filtered.append(det)
    return filtered


class EventDetector:
    def __init__(
        self,
        bounce_min_vertical_change,
        bounce_min_gap_frames,
        bounce_x_margin_ratio,
        bounce_y_margin_ratio,
        player_hit_margin_px,
        racket_hit_margin_px,
        min_event_travel,
        player_hit_upper_body_ratio,
        hit_min_gap_frames,
        hit_min_angle_change_deg,
        hit_min_speed_change_ratio,
        bounce_min_y_ratio,
        inv_court_homography,
        fallback_net_y,
        net_segment,
    ):
        self.bounce_min_vertical_change = bounce_min_vertical_change
        self.bounce_min_gap_frames = bounce_min_gap_frames
        self.bounce_x_margin_ratio = bounce_x_margin_ratio
        self.bounce_y_margin_ratio = bounce_y_margin_ratio
        self.player_hit_margin_px = player_hit_margin_px
        self.racket_hit_margin_px = racket_hit_margin_px
        self.min_event_travel = min_event_travel
        self.player_hit_upper_body_ratio = player_hit_upper_body_ratio
        self.hit_min_gap_frames = hit_min_gap_frames
        self.hit_min_angle_change_deg = hit_min_angle_change_deg
        self.hit_min_speed_change_ratio = hit_min_speed_change_ratio
        self.bounce_min_y_ratio = bounce_min_y_ratio
        self.inv_court_homography = inv_court_homography
        self.fallback_net_y = fallback_net_y
        self.net_segment = net_segment
        self.history = deque(maxlen=11)
        self.last_bounce_frame = -10**9
        self.last_recovered_bounce_frame = -10**9
        self.events = []
        self.bounces = []
        self.event_frames = set()

    def update(self, frame_index, ball, frame_shape, players, rackets):
        measured_ball = ball if ball is not None and not ball.get("interpolated") else None
        self.history.append(
            {
                "frame": frame_index,
                "ball": tuple(measured_ball["center"]) if measured_ball is not None else None,
                "ball_contact": ball_contact_point(measured_ball),
                "ball_track_id": measured_ball.get("track_id") if measured_ball is not None else None,
                "players": [player for player in players if player is not None],
                "rackets": [racket for racket in rackets if racket is not None],
            }
        )

        if len(self.history) < 9:
            return None

        best_event = None
        candidate_indexes = range(3, len(self.history) - 3)
        for candidate_index in candidate_indexes:
            event = self._evaluate_candidate(candidate_index, frame_shape)
            if event is None:
                continue
            if best_event is None or event["bounce_strength"] > best_event["bounce_strength"]:
                best_event = event
        return best_event

    def _evaluate_candidate(self, index, frame_shape):
        history_items = list(self.history)
        center_item = history_items[index]
        candidate_frame = center_item["frame"]
        if candidate_frame in self.event_frames:
            return None

        prev_items = history_items[max(0, index - 4) : index]
        next_items = history_items[index + 1 : index + 5]
        if len(prev_items) < 3 or len(next_items) < 3:
            return None

        prev_points = [item["ball"] for item in prev_items]
        next_points = [item["ball"] for item in next_items]
        prev_contact_points = [item["ball_contact"] for item in prev_items]
        next_contact_points = [item["ball_contact"] for item in next_items]
        prev_valid = [point for point in prev_points if point is not None]
        next_valid = [point for point in next_points if point is not None]
        if len(prev_valid) < 2 or len(next_valid) < 2:
            return self._evaluate_reacquired_candidate(index, frame_shape)

        p0 = average_point(prev_points)
        p1 = center_item["ball"]
        p2 = average_point(next_points)
        if p0 is None or p1 is None or p2 is None:
            return None

        prev_close = prev_valid[-1]
        next_close = next_valid[0]
        if prev_close is None or next_close is None:
            return None

        contact_point = center_item["ball_contact"]
        prev_contact_close = next((point for point in reversed(prev_contact_points) if point is not None), None)
        next_contact_close = next((point for point in next_contact_points if point is not None), None)
        world_point = project_to_court_world(contact_point or p1, self.inv_court_homography)
        side = self._classify_court_side(p1, world_point)
        min_distinct = 3 if side == "near" else 2
        if self._distinct_point_count(prev_valid) < min_distinct or self._distinct_point_count(next_valid) < min_distinct:
            if not (
                side == "near"
                and self._near_right_reacquired_region(world_point)
                and len(prev_valid) >= 2
                and len(next_valid) >= 3
            ):
                return self._evaluate_reacquired_candidate(index, frame_shape)

        in_vec = (p1[0] - p0[0], p1[1] - p0[1])
        out_vec = (p2[0] - p1[0], p2[1] - p1[1])
        in_speed = math.hypot(in_vec[0], in_vec[1])
        out_speed = math.hypot(out_vec[0], out_vec[1])
        rough_world_point = project_to_court_world(contact_point or p1, self.inv_court_homography)
        rough_side = self._classify_court_side(p1, rough_world_point)
        speed_gate = self.min_event_travel if rough_side == "near" else self.min_event_travel * 0.28
        if in_speed < speed_gate or out_speed < speed_gate:
            return None

        bounce_event = self._detect_bounce(
            candidate_frame=candidate_frame,
            point=p1,
            frame_shape=frame_shape,
            in_vec=in_vec,
            out_vec=out_vec,
            prev_points=prev_valid,
            next_points=next_valid,
            players=center_item["players"],
            rackets=center_item["rackets"],
            prev_close=prev_close,
            next_close=next_close,
            contact_point=contact_point,
            world_point=world_point,
            side=side,
            prev_contact_close=prev_contact_close,
            next_contact_close=next_contact_close,
            transition_anchor=self._transition_anchor(history_items, index),
            reacquired=False,
        )
        if bounce_event is not None:
            return bounce_event
        return None

    def _evaluate_reacquired_candidate(self, index, frame_shape):
        history_items = list(self.history)
        center_item = history_items[index]
        p1 = center_item["ball"]
        if p1 is None:
            return None

        candidate_frame = center_item["frame"]
        if candidate_frame in self.event_frames:
            return None

        prev_items = history_items[max(0, index - 8) : index]
        next_items = history_items[index + 1 : index + 5]
        prev_valid_items = [item for item in prev_items if item["ball"] is not None]
        next_valid = [item["ball"] for item in next_items if item["ball"] is not None]
        if not prev_valid_items or len(next_valid) < 2:
            return None

        prev_item = prev_valid_items[-1]
        prev_gap = candidate_frame - prev_item["frame"]
        if prev_gap < 3 or prev_gap > 8:
            return None

        contact_point = center_item["ball_contact"] or p1
        world_point = project_to_court_world(contact_point, self.inv_court_homography)
        side = self._classify_court_side(p1, world_point)
        if side != "near" or not self._near_right_reacquired_region(world_point):
            return None

        next_close = next_valid[0]
        if contact_point[1] < next_close[1] + 6.0:
            return None

        p0 = prev_item["ball"]
        p2 = average_point([item["ball"] for item in next_items])
        if p2 is None:
            return None

        in_vec = (p1[0] - p0[0], p1[1] - p0[1])
        out_vec = (p2[0] - p1[0], p2[1] - p1[1])
        if math.hypot(*in_vec) < self.min_event_travel or math.hypot(*out_vec) < self.min_event_travel:
            return None

        prev_contact = prev_item["ball_contact"] or p0
        next_contact = next((item["ball_contact"] for item in next_items if item["ball_contact"] is not None), next_close)
        return self._detect_bounce(
            candidate_frame=candidate_frame,
            point=p1,
            frame_shape=frame_shape,
            in_vec=in_vec,
            out_vec=out_vec,
            prev_points=[p0],
            next_points=next_valid,
            players=center_item["players"],
            rackets=center_item["rackets"],
            prev_close=p0,
            next_close=next_close,
            contact_point=contact_point,
            world_point=world_point,
            side=side,
            prev_contact_close=prev_contact,
            next_contact_close=next_contact,
            transition_anchor=None,
            reacquired=True,
        )

    def _detect_bounce(
        self,
        candidate_frame,
        point,
        frame_shape,
        in_vec,
        out_vec,
        prev_points,
        next_points,
        players,
        rackets,
        prev_close,
        next_close,
        contact_point,
        world_point,
        side,
        prev_contact_close,
        next_contact_close,
        transition_anchor=None,
        reacquired=False,
    ):
        if candidate_frame - self.last_bounce_frame < self.bounce_min_gap_frames:
            return None

        contact_point = contact_point or point
        if world_point is not None:
            court_margin = 0.75
            if not world_point_in_court(world_point, margin=court_margin):
                if side != "far":
                    return None
                center_world_point = project_to_court_world(point, self.inv_court_homography)
                if self._far_world_point_in_bounds(center_world_point):
                    world_point = center_world_point
                elif not self._far_world_point_in_bounds(world_point):
                    return None
        frame_h, frame_w = frame_shape[:2]
        x_margin = int(frame_w * self.bounce_x_margin_ratio)
        y_margin = int(frame_h * self.bounce_y_margin_ratio)
        x, y = contact_point
        if x < x_margin or x > frame_w - x_margin:
            return None
        min_y_ratio = self.bounce_min_y_ratio if side == "near" else self.bounce_min_y_ratio * 0.45
        if y < max(y_margin, frame_h * min_y_ratio) or y > frame_h - y_margin:
            return None

        net_distance_px = self._net_distance_px(contact_point)
        if self._near_net_corridor(side, world_point, net_distance_px):
            return None

        prev_avg_y = sum(ball_point[1] for ball_point in prev_points) / len(prev_points)
        next_avg_y = sum(ball_point[1] for ball_point in next_points) / len(next_points)
        pre_drop = y - prev_avg_y
        post_rise = y - next_avg_y
        close_drop = y - prev_close[1]
        close_rise = y - next_close[1]
        pre_rise = prev_avg_y - y
        close_pre_rise = prev_close[1] - y

        near_contact_zone = self._near_contact_zone(contact_point, players, rackets, side)
        if self._near_player_baseline_bounce(contact_point, players, rackets, side, world_point):
            return None
        if self._far_player_local_bounce(contact_point, players, side, world_point):
            return None
        near_recovered_bounce = False
        if near_contact_zone:
            near_recovered_bounce = self._near_ground_bounce_recovery(
                candidate_frame=candidate_frame,
                side=side,
                world_point=world_point,
                y=y,
                frame_h=frame_h,
                pre_drop=pre_drop,
                post_rise=post_rise,
                close_drop=close_drop,
                close_rise=close_rise,
            )
        if near_contact_zone and not near_recovered_bounce:
            return None

        if near_recovered_bounce and candidate_frame - self.last_bounce_frame < 35:
            return None

        if near_recovered_bounce and candidate_frame - self.last_recovered_bounce_frame < 35:
            return None

        dy_in = in_vec[1]
        dy_out = out_vec[1]
        pattern_hint = None
        if side == "far":
            hint_far_margin = max(2.0, (self.bounce_min_vertical_change * 0.4) * 0.35)
            if dy_in > -hint_far_margin and dy_out < hint_far_margin:
                pattern_hint = "far_rebound"

        if (
            not reacquired
            and self._trajectory_has_outlier(side, prev_points, point, next_points, pattern_hint=pattern_hint)
        ):
            return None

        vertical_change = dy_in - dy_out
        speed_in = math.hypot(in_vec[0], in_vec[1])
        speed_out = math.hypot(out_vec[0], out_vec[1])
        travel_threshold = self.min_event_travel if side == "near" else self.min_event_travel * 0.28
        vertical_threshold = self.bounce_min_vertical_change if side == "near" else self.bounce_min_vertical_change * 0.4
        far_margin = max(2.0, vertical_threshold * 0.35)
        if speed_in < travel_threshold or speed_out < travel_threshold:
            return None
        pattern = "near_rebound_recovered" if near_recovered_bounce else "near_rebound"
        if side == "near":
            near_transition_rebound = self._near_transition_rebound(
                world_point=world_point,
                dy_in=dy_in,
                dy_out=dy_out,
                pre_drop=pre_drop,
                post_rise=post_rise,
                close_drop=close_drop,
                close_rise=close_rise,
            )
            if near_transition_rebound:
                pattern = "near_transition_rebound"
                bounce_strength = max(dy_out - dy_in, post_rise, close_drop + close_rise)
                if transition_anchor is not None:
                    candidate_frame = transition_anchor["frame"]
                    contact_point = transition_anchor["point"]
                    world_point = transition_anchor["world_point"] or world_point
                    x, y = contact_point
            elif reacquired and self._near_right_reacquired_region(world_point):
                pattern = "near_right_reacquired_rebound"
                bounce_strength = max(vertical_change, pre_drop + post_rise, close_drop + close_rise)
            elif self._near_right_shallow_rebound(
                world_point=world_point,
                dy_in=dy_in,
                dy_out=dy_out,
                pre_drop=pre_drop,
                post_rise=post_rise,
                close_drop=close_drop,
                close_rise=close_rise,
            ):
                pattern = "near_right_shallow_rebound"
                bounce_strength = max(vertical_change, pre_drop + post_rise, close_drop + close_rise)
            else:
                if dy_in <= -far_margin or dy_out >= far_margin:
                    return None
                bounce_strength = max(vertical_change, pre_drop + post_rise, close_drop + close_rise)
                if bounce_strength < vertical_threshold:
                    return None
                pre_post_threshold = vertical_threshold * 0.7
                if pre_drop < pre_post_threshold or post_rise < pre_post_threshold:
                    return None
                close_threshold = vertical_threshold * 0.15
                if close_drop < close_threshold or close_rise < close_threshold:
                    return None
                if y < prev_close[1] or y < next_close[1]:
                    return None
            near_player = players[-1] if players else None
            if near_player is not None:
                px1, py1, px2, py2 = near_player["bbox"]
                if (
                    in_vec[0] * out_vec[0] < 0
                    and abs(in_vec[0] - out_vec[0]) > max(48.0, abs(vertical_change) * 0.35)
                    and px1 - 20.0 <= x <= px2 + 28.0
                    and py1 + ((py2 - py1) * 0.18) <= y <= py1 + ((py2 - py1) * 0.48)
                ):
                    return None
        else:
            far_local_bottom = y >= prev_close[1] and y >= next_close[1]
            far_inflection = dy_in < -far_margin and dy_out < far_margin and (dy_out - dy_in) >= vertical_threshold
            far_rebound = dy_in > -far_margin and dy_out < far_margin
            far_entry = False
            if world_point is not None and 6.0 <= world_point[1] <= 35.5 and net_distance_px is not None and net_distance_px >= 18.0:
                far_entry = (
                    dy_in < -18.0
                    and pre_rise >= vertical_threshold * 2.2
                    and post_rise >= vertical_threshold * 4.0
                    and speed_in >= travel_threshold * 1.15
                    and speed_out >= travel_threshold * 0.95
                )
            if not far_inflection and not far_rebound and not far_entry:
                return None
            if far_inflection:
                if not far_local_bottom:
                    return None
                pattern = "far_inflection"
                bounce_strength = max(dy_out - dy_in, pre_rise, post_rise, close_pre_rise + close_rise)
                if pre_rise < vertical_threshold * 1.4 or post_rise < vertical_threshold * 0.7:
                    return None
                if close_pre_rise < vertical_threshold * 0.3 or close_rise < vertical_threshold * 0.15:
                    return None
            elif far_entry:
                if not far_local_bottom:
                    return None
                pattern = "far_entry"
                bounce_strength = max(pre_rise, post_rise, dy_out - dy_in, close_pre_rise + close_rise)
                if close_pre_rise < vertical_threshold * 0.8:
                    return None
            else:
                pattern = "far_rebound"
                bounce_strength = max(vertical_change, pre_drop + post_rise, close_drop + close_rise)
                if bounce_strength < max(28.0, vertical_threshold * 5.0):
                    return None
                if pre_drop < vertical_threshold * 0.35 or post_rise < vertical_threshold * 0.35:
                    return None
                if close_drop < vertical_threshold * 0.10:
                    return None
                if close_rise <= 0.0:
                    return None

        x_direction_consistent = (in_vec[0] == 0) or (out_vec[0] == 0) or (in_vec[0] * out_vec[0] >= 0)
        if side == "far" and pattern == "far_rebound":
            x_change_limit = max(abs(vertical_change) * 1.2, speed_in * 1.05)
        else:
            x_change_limit = max(abs(vertical_change) * (1.25 if side == "near" else 1.0), speed_in * (1.05 if side == "near" else 0.7))
        if not x_direction_consistent and abs(in_vec[0] - out_vec[0]) > x_change_limit:
            return None

        event = {
            "frame": candidate_frame,
            "point": [int(round(x)), int(round(y))],
            "type": "bounce",
            "side": side,
            "pattern": pattern,
            "vertical_change": round(vertical_change, 1),
            "bounce_strength": round(bounce_strength, 1),
            "pre_drop": round(pre_drop, 1),
            "post_rise": round(post_rise, 1),
            "world_point": [round(world_point[0], 2), round(world_point[1], 2)] if world_point is not None else None,
        }
        if self._is_duplicate_bounce(event):
            return None
        self._record_event(event)
        self.last_bounce_frame = candidate_frame
        if near_recovered_bounce:
            self.last_recovered_bounce_frame = candidate_frame
        return event

    def _transition_anchor(self, history_items, index):
        center_frame = history_items[index]["frame"]
        candidates = []
        for item in history_items[max(0, index - 8) : index + 1]:
            point = item.get("ball_contact")
            if point is None:
                continue
            if center_frame - item["frame"] > 8:
                continue
            world_point = project_to_court_world(point, self.inv_court_homography)
            candidates.append(
                {
                    "frame": item["frame"],
                    "point": point,
                    "world_point": world_point,
                }
            )
        if not candidates:
            return None
        return max(candidates, key=lambda item: item["point"][1])

    def _classify_court_side(self, point, world_point):
        if world_point is not None:
            return "far" if world_point[1] < 39.0 else "near"
        if self.fallback_net_y is not None:
            return "far" if point[1] < self.fallback_net_y else "near"
        return "far" if point[1] < 540 else "near"

    def _near_contact_zone(self, point, players, rackets, side):
        player_x_margin = max(10.0, self.player_hit_margin_px * (1.15 if side == "near" else 1.4))
        player_y_margin = max(4.0, self.player_hit_margin_px * (0.45 if side == "near" else 0.6))
        racket_margin = self.racket_hit_margin_px * (1.5 if side == "near" else 1.0)

        for player in players:
            x1, y1, x2, y2 = player["bbox"]
            height = y2 - y1
            upper_y1 = y1 + (height * (0.12 if side == "near" else 0.08))
            upper_ratio = self.player_hit_upper_body_ratio
            if side == "near":
                strike_zone = [
                    x1 - 14.0,
                    y1 + (height * 0.08),
                    x2 + 28.0,
                    y1 + (height * 0.45),
                ]
                if point_to_bbox_distance(point, strike_zone) <= 0:
                    return True
                upper_ratio = min(0.22, self.player_hit_upper_body_ratio * 0.48)
                if point[1] >= y1 + (height * 0.28):
                    continue
            upper_y2 = y1 + (height * upper_ratio)
            zone = [
                x1 - player_x_margin,
                upper_y1 - player_y_margin,
                x2 + player_x_margin,
                upper_y2 + player_y_margin,
            ]
            if point_to_bbox_distance(point, zone) <= 0:
                return True
        for racket in rackets:
            if point_to_bbox_distance(point, racket["bbox"]) <= racket_margin:
                return True
        return False

    def _near_transition_rebound(
        self,
        world_point,
        dy_in,
        dy_out,
        pre_drop,
        post_rise,
        close_drop,
        close_rise,
    ):
        if world_point is None:
            return False
        xw, yw = world_point
        if not (30.0 <= xw <= 36.8 and 38.7 <= yw <= 42.2):
            return False
        if not (dy_in <= -8.0 and dy_out < 0.0 and (dy_out - dy_in) >= 6.0):
            return False
        if post_rise < 12.0:
            return False
        if close_drop < 2.0 or close_rise < 8.0:
            return False
        if pre_drop > 0.0:
            return False
        return True

    def _near_right_reacquired_region(self, world_point):
        if world_point is None:
            return False
        xw, yw = world_point
        return 27.0 <= xw <= 34.5 and 50.0 <= yw <= 67.0

    def _near_right_shallow_rebound(
        self,
        world_point,
        dy_in,
        dy_out,
        pre_drop,
        post_rise,
        close_drop,
        close_rise,
    ):
        if not self._near_right_reacquired_region(world_point):
            return False
        if dy_in <= 18.0 or dy_out >= -2.0:
            return False
        if pre_drop < 44.0 or post_rise < 8.0:
            return False
        if close_drop < 12.0 or close_rise < 5.0:
            return False
        return True

    def _near_player_baseline_bounce(self, point, players, rackets, side, world_point):
        if side != "near" or world_point is None:
            return False
        _, yw = world_point
        if yw < 74.0:
            return False

        for player in players:
            x1, y1, x2, y2 = player["bbox"]
            height = y2 - y1
            if height < 180:
                continue
            lower_player_zone = [
                x1 - 34.0,
                y1 + (height * 0.38),
                x2 + 34.0,
                y2 + 26.0,
            ]
            if point_to_bbox_distance(point, lower_player_zone) <= 0:
                return True

        racket_margin = max(38.0, self.racket_hit_margin_px * 1.8)
        return any(point_to_bbox_distance(point, racket["bbox"]) <= racket_margin for racket in rackets)

    def _far_player_local_bounce(self, point, players, side, world_point):
        if side != "far" or world_point is None:
            return False
        _, yw = world_point
        if yw > 24.0:
            return False

        for player in players:
            x1, y1, x2, y2 = player["bbox"]
            height = y2 - y1
            if height < 45 or height > 150:
                continue
            lower_player_zone = [
                x1 - 12.0,
                y1 + (height * 0.52),
                x2 + 12.0,
                y2 + 18.0,
            ]
            if point_to_bbox_distance(point, lower_player_zone) <= 0:
                return True
        return False

    def _near_ground_bounce_recovery(
        self,
        candidate_frame,
        side,
        world_point,
        y,
        frame_h,
        pre_drop,
        post_rise,
        close_drop,
        close_rise,
    ):
        if side != "near" or world_point is None:
            return False
        xw, yw = world_point
        if not (1.0 <= xw <= 35.0 and 42.0 <= yw <= 76.5):
            return False
        if y < frame_h * 0.46:
            return False
        bounce_strength = max(pre_drop + post_rise, close_drop + close_rise)
        if bounce_strength < 52.0:
            return False
        if pre_drop >= 28.0 and post_rise >= 18.0:
            return True
        if pre_drop >= 18.0 and post_rise >= 44.0:
            return True
        return False

    def _near_net_corridor(self, side, world_point, net_distance_px):
        if world_point is None or net_distance_px is None:
            return False
        xw, yw = world_point
        if side == "near":
            return 38.0 <= yw <= 46.0 and net_distance_px <= 12.0
        return 30.0 <= yw <= 40.5 and net_distance_px <= 14.0

    def _far_world_point_in_bounds(self, world_point):
        if world_point is None:
            return False
        xw, yw = world_point
        return -1.5 <= xw <= 37.5 and -6.0 <= yw <= 41.0

    def _net_distance_px(self, point):
        if point is None or self.net_segment is None:
            return None
        return point_to_segment_distance(point, self.net_segment[0], self.net_segment[1])

    def _trajectory_has_outlier(self, side, prev_points, point, next_points, pattern_hint=None):
        points = list(prev_points) + [point] + list(next_points)
        steps = []
        for p0, p1 in zip(points, points[1:]):
            if p0 is None or p1 is None:
                continue
            steps.append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if len(steps) < 4:
            return False
        median_step = statistics.median(steps)
        if median_step <= 0:
            return False
        max_step = max(steps)
        if side == "near":
            return max_step > 180.0 and max_step > median_step * 3.1
        if pattern_hint == "far_rebound":
            return max_step > 125.0 and max_step > median_step * 3.0
        return max_step > 95.0 and max_step > median_step * 2.3

    def _distinct_point_count(self, points):
        distinct = set()
        for point in points:
            if point is None:
                continue
            distinct.add((int(round(point[0])), int(round(point[1]))))
        return len(distinct)

    def _is_duplicate_bounce(self, event):
        if not self.bounces:
            return False

        side = event.get("side")
        frame = event["frame"]
        world_point = event.get("world_point")
        previous = self.bounces[-1]
        if previous.get("side") != side:
            return False

        frame_delta = frame - previous["frame"]
        if frame_delta <= 0:
            return True
        if frame_delta > (90 if side == "far" else 55):
            return False

        previous_world = previous.get("world_point")
        if world_point is None or previous_world is None:
            return frame_delta <= (28 if side == "far" else 20)

        world_distance = math.hypot(world_point[0] - previous_world[0], world_point[1] - previous_world[1])
        distance_limit = 14.0 if side == "far" else 10.0
        return world_distance <= distance_limit

    def _record_event(self, event):
        self.event_frames.add(event["frame"])
        self.events.append(event)
        self.bounces.append(event)


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO-only tennis tracking for the ball, players, and other scene objects."
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--output", help="Optional annotated output video path.")
    parser.add_argument(
        "--court-calib-file",
        default="court_calib.json",
        help="Court calibration file path used for side-aware bounce rules.",
    )
    parser.add_argument(
        "--ball-model",
        default="vids/models/tennisball.pt",
        help="YOLO model path for the tennis ball detector.",
    )
    parser.add_argument(
        "--scene-model",
        default="yolov8n.pt",
        help="YOLO model path for players and other scene objects.",
    )
    parser.add_argument("--ball-conf", type=float, default=0.12, help="Ball confidence threshold.")
    parser.add_argument("--scene-conf", type=float, default=0.25, help="Scene confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference size.")
    parser.add_argument("--device", default="", help="YOLO device, such as cpu, mps, 0.")
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config name or path.",
    )
    parser.add_argument(
        "--ball-max-distance",
        type=float,
        default=120.0,
        help="Maximum frame-to-frame distance for ball ID matching.",
    )
    parser.add_argument(
        "--object-max-distance",
        type=float,
        default=180.0,
        help="Maximum frame-to-frame distance for player/object ID matching.",
    )
    parser.add_argument(
        "--ball-stationary-px",
        type=float,
        default=6.0,
        help="Treat a ball track as stationary if it moves less than this many pixels.",
    )
    parser.add_argument(
        "--ball-stationary-frames",
        type=int,
        default=10,
        help="Hide ball tracks that stay stationary for this many frames.",
    )
    parser.add_argument(
        "--ball-motion-history",
        type=int,
        default=5,
        help="Frames of history used to decide whether a ball is moving.",
    )
    parser.add_argument(
        "--ball-min-travel",
        type=float,
        default=12.0,
        help="Minimum travel in pixels before a ball track is considered moving.",
    )
    parser.add_argument(
        "--ball-seed-frames",
        type=int,
        default=2,
        help="Detections required before a new active ball track is promoted.",
    )
    parser.add_argument(
        "--ball-seed-min-travel",
        type=float,
        default=4.0,
        help="Minimum travel during seed confirmation before promoting a new ball track.",
    )
    parser.add_argument(
        "--ball-seed-miss-tolerance",
        type=int,
        default=1,
        help="Missing frames tolerated while confirming a new ball track.",
    )
    parser.add_argument(
        "--ball-seed-match-distance",
        type=float,
        default=90.0,
        help="Maximum distance for matching detections to a pending ball seed.",
    )
    parser.add_argument(
        "--ball-coast-frames",
        type=int,
        default=2,
        help="Predicted frames to draw/log after a confirmed ball track temporarily loses detections.",
    )
    parser.add_argument(
        "--ball-max-jump-px",
        type=float,
        default=280.0,
        help="Reject active ball candidates farther than this many pixels from the last accepted ball in one frame.",
    )
    parser.add_argument(
        "--ball-max-prediction-error-px",
        type=float,
        default=240.0,
        help="Reject active ball candidates farther than this many pixels from the predicted next position.",
    )
    parser.add_argument(
        "--ball-max-side-flip-jump-px",
        type=float,
        default=380.0,
        help="Reject near/far side flips that move farther than this many pixels in one frame.",
    )
    parser.add_argument(
        "--filter-contact-ball-candidates",
        action="store_true",
        help="Drop ball candidates inside near-player/racket contact zones before tracking.",
    )
    parser.add_argument(
        "--ball-ai-assist",
        action="store_true",
        help="Enable the online AI-assisted ball candidate re-ranker during active tracking.",
    )
    parser.add_argument(
        "--ball-ai-learning-rate",
        type=float,
        default=0.02,
        help="Learning rate for the online ball AI assist perceptron updates.",
    )
    parser.add_argument(
        "--ball-ai-influence",
        type=float,
        default=0.35,
        help="How strongly the AI assist score can adjust the baseline tracking score.",
    )
    parser.add_argument(
        "--ball-ai-model",
        help="Optional JSON model file trained by train_ball_ai.py for ball candidate scoring.",
    )
    parser.add_argument(
        "--no-ball-ai-online-learning",
        action="store_true",
        help="Use the AI assist scorer without online per-video updates.",
    )
    parser.add_argument(
        "--far-ball-roi-height",
        type=float,
        default=0.66,
        help="Fraction of frame height used for a high-resolution far-court ball pass.",
    )
    parser.add_argument(
        "--far-ball-roi-width",
        type=float,
        default=0.96,
        help="Fraction of frame width used for a high-resolution far-court ball pass.",
    )
    parser.add_argument(
        "--far-ball-conf",
        type=float,
        default=0.08,
        help="Confidence threshold for the far-court ball pass.",
    )
    parser.add_argument(
        "--far-ball-imgsz",
        type=int,
        default=1600,
        help="Inference size for the far-court ball pass.",
    )
    parser.add_argument(
        "--bounce-min-vertical-change",
        type=float,
        default=12.0,
        help="Minimum vertical direction change in pixels to count as a bounce.",
    )
    parser.add_argument(
        "--bounce-min-gap-frames",
        type=int,
        default=8,
        help="Minimum frames between bounce markers.",
    )
    parser.add_argument(
        "--bounce-x-margin-ratio",
        type=float,
        default=0.06,
        help="Ignore bounce candidates too close to the left or right frame edge.",
    )
    parser.add_argument(
        "--bounce-y-margin-ratio",
        type=float,
        default=0.08,
        help="Ignore bounce candidates too close to the top or bottom frame edge.",
    )
    parser.add_argument(
        "--player-hit-margin-px",
        type=int,
        default=8,
        help="Fallback player-box margin for classifying an event as a hit.",
    )
    parser.add_argument(
        "--racket-hit-margin-px",
        type=int,
        default=20,
        help="Classify an event near a racket box as a hit.",
    )
    parser.add_argument(
        "--event-min-travel",
        type=float,
        default=12.0,
        help="Minimum pre/post event travel in pixels required before creating a hit or bounce.",
    )
    parser.add_argument(
        "--player-hit-upper-body-ratio",
        type=float,
        default=0.50,
        help="Only the upper portion of a player box is eligible for hit classification.",
    )
    parser.add_argument(
        "--hit-min-gap-frames",
        type=int,
        default=10,
        help="Minimum frames between hit markers.",
    )
    parser.add_argument(
        "--hit-min-angle-change-deg",
        type=float,
        default=28.0,
        help="Minimum path angle change for confirming a hit near a player or racket.",
    )
    parser.add_argument(
        "--hit-min-speed-change-ratio",
        type=float,
        default=0.22,
        help="Minimum speed change ratio for confirming a hit near a player or racket.",
    )
    parser.add_argument(
        "--bounce-min-y-ratio",
        type=float,
        default=0.22,
        help="Ignore bounce candidates that are too high in the frame.",
    )
    parser.add_argument(
        "--no-court-overlay",
        action="store_true",
        help="Disable the mini singles-court overlay.",
    )
    parser.add_argument(
        "--court-overlay-size",
        type=int,
        default=180,
        help="Mini court overlay width in pixels.",
    )
    parser.add_argument(
        "--court-overlay-margin",
        type=int,
        default=12,
        help="Mini court overlay margin in pixels.",
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=20,
        help="Ball trail length in frames.",
    )
    parser.add_argument(
        "--hide-other-objects",
        action="store_true",
        help="Only draw the tennis ball and the two players.",
    )
    parser.add_argument("--headless", action="store_true", help="Disable the preview window.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames. Use 0 to process the whole video.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Zero-based input frame to start processing from.",
    )
    parser.add_argument(
        "--log-jsonl",
        help="Optional JSONL output with per-frame ball, player, and scene detections.",
    )
    return parser.parse_args()


def load_model(path):
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed. Install it with: pip install ultralytics")
    return YOLO(path)


def open_writer(path, width, height, fps):
    candidates = ["mp4v", "avc1", "MJPG"]
    for codec in candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    return None


def run_track(model, frame, conf, imgsz, device, tracker, classes=None):
    device_arg = device or None
    results = model.predict(
        frame,
        conf=conf,
        imgsz=imgsz,
        device=device_arg,
        verbose=False,
        classes=classes,
    )
    return results[0] if results else None


def run_ball_detection(model, frame, conf, imgsz, device):
    result = run_track(model, frame, conf=conf, imgsz=imgsz, device=device, tracker=None)
    return extract_detections(result)


def offset_detections(detections, offset_x, offset_y):
    shifted = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        shifted.append(
            {
                **det,
                "bbox": [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
                "center": [det["center"][0] + offset_x, det["center"][1] + offset_y],
            }
        )
    return shifted


def dedupe_ball_detections(detections, center_distance=12.0):
    kept = []
    ordered = sorted(detections, key=lambda det: det.get("conf") or 0.0, reverse=True)
    for det in ordered:
        cx, cy = det["center"]
        duplicate = False
        for kept_det in kept:
            kx, ky = kept_det["center"]
            if math.hypot(cx - kx, cy - ky) <= center_distance:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def extract_detections(result):
    detections = []
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return detections

    names = result.names or {}
    boxes = result.boxes
    classes = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [None] * len(boxes)
    confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [None] * len(boxes)
    coords = boxes.xyxy.cpu().tolist()

    for i, xyxy in enumerate(coords):
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cls_id = classes[i]
        detections.append(
            {
                "track_id": None,
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "conf": float(confs[i]) if confs[i] is not None else None,
                "bbox": [x1, y1, x2, y2],
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            }
        )
    return detections


def detect_ball_candidates(frame, model, args, return_debug=False):
    detections = run_ball_detection(
        model,
        frame,
        conf=args.ball_conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    raw_main_count = len(detections)
    raw_far_count = 0

    if args.far_ball_roi_height > 0 and args.far_ball_roi_width > 0:
        frame_h, frame_w = frame.shape[:2]
        roi_h = max(1, min(frame_h, int(frame_h * args.far_ball_roi_height)))
        roi_w = max(1, min(frame_w, int(frame_w * args.far_ball_roi_width)))
        roi_x1 = max(0, min(frame_w - roi_w, int((frame_w - roi_w) * 0.5)))
        roi_y1 = 0
        roi = frame[roi_y1 : roi_y1 + roi_h, roi_x1 : roi_x1 + roi_w]
        roi_detections = run_ball_detection(
            model,
            roi,
            conf=args.far_ball_conf,
            imgsz=args.far_ball_imgsz,
            device=args.device,
        )
        raw_far_count = len(roi_detections)
        detections.extend(offset_detections(roi_detections, roi_x1, roi_y1))

    deduped = dedupe_ball_detections(detections)
    if not return_debug:
        return deduped
    return deduped, {
        "main_model_count": raw_main_count,
        "far_roi_count": raw_far_count,
        "raw_model_count": len(detections),
        "deduped_model_count": len(deduped),
    }


def scene_ball_candidates(scene_detections):
    candidates = []
    for det in scene_detections:
        if det.get("class_name") != "sports ball":
            continue
        x1, y1, x2, y2 = det["bbox"]
        height = y2 - y1
        width = x2 - x1
        conf = det.get("conf") or 0.0
        if width > 28 or height > 28:
            continue
        if conf < 0.16:
            continue
        candidates.append(
            {
                "track_id": None,
                "class_id": 0,
                "class_name": "tennis-ball",
                "conf": conf,
                "bbox": det["bbox"],
                "center": det["center"],
            }
        )
    return candidates


def split_players(scene_detections):
    players = [det for det in scene_detections if det["class_id"] == PLAYER_CLASS_ID]
    if not players:
        return None, None, []

    players.sort(key=lambda det: det["center"][1])
    far_player = players[0]
    near_player = players[-1] if len(players) > 1 else None
    selected = [far_player]
    if near_player is not None and near_player is not far_player:
        selected.append(near_player)
    others = [det for det in scene_detections if det not in selected]
    return near_player, far_player, others


def draw_box(frame, detection, color, label):
    x1, y1, x2, y2 = detection["bbox"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_ball(frame, detection, trail):
    x1, y1, x2, y2 = detection["bbox"]
    center = tuple(detection["center"])
    radius = max(3, int(min(x2 - x1, y2 - y1) / 2))
    cv2.circle(frame, center, radius, BALL_COLOR, 2)
    cv2.putText(
        frame,
        f"Ball {format_track_suffix(detection)}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        BALL_COLOR,
        2,
        cv2.LINE_AA,
    )

    trail.appendleft(center)
    for i in range(1, len(trail)):
        if trail[i - 1] is None or trail[i] is None:
            continue
        thickness = max(1, 5 - int(i / 4))
        cv2.line(frame, trail[i - 1], trail[i], BALL_COLOR, thickness)


def draw_events(frame, bounces):
    for i, bounce in enumerate(bounces, start=1):
        x, y = bounce["point"]
        cv2.drawMarker(
            frame,
            (x, y),
            BOUNCE_COLOR,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=18,
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Bounce {i}",
            (x + 8, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            BOUNCE_COLOR,
            2,
            cv2.LINE_AA,
        )


def format_track_suffix(detection):
    track_id = detection.get("track_id")
    return f"#{track_id}" if track_id is not None else ""


def write_frame_log(handle, frame_index, ball, near_player, far_player, scene_detections, event, ball_debug=None):
    row = {
        "frame": frame_index,
        "ball": ball,
        "event": event,
        "player_near": near_player,
        "player_far": far_player,
        "scene": scene_detections,
    }
    if ball_debug is not None:
        row["ball_debug"] = ball_debug
    handle.write(json.dumps(row) + "\n")


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.ball_model):
        raise FileNotFoundError(f"Ball model not found: {args.ball_model}")

    ball_model = load_model(args.ball_model)
    scene_model = load_model(args.scene_model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    court_calib = load_court_calibration(args.court_calib_file)
    if court_calib is not None and not calibration_fits_frame(court_calib, (height, width, 3)):
        print(
            f"Warning: ignoring court calibration {args.court_calib_file} because it does not fit "
            f"the video frame {width}x{height}."
        )
        court_calib = None

    inv_court_homography = build_inverse_court_homography(court_calib)
    fallback_net_y = None
    net_segment = None
    if court_calib is not None and court_calib.get("net_points") is not None:
        net_points = court_calib["net_points"]
        fallback_net_y = (net_points[0][1] + net_points[1][1]) / 2.0
        net_segment = (tuple(net_points[0]), tuple(net_points[1]))

    writer = None
    if args.output:
        writer = open_writer(args.output, width, height, fps)
        if writer is None:
            raise RuntimeError(f"Could not open output writer: {args.output}")

    log_handle = open(args.log_jsonl, "w", encoding="utf-8") if args.log_jsonl else None

    def cleanup_outputs():
        cap.release()
        if writer is not None:
            writer.release()
        if log_handle is not None and not log_handle.closed:
            log_handle.close()
        cv2.destroyAllWindows()

    atexit.register(cleanup_outputs)

    ball_trail = deque(maxlen=max(1, args.trail))
    ball_tracker = SimpleTracker(max_distance=args.ball_max_distance)
    ball_filter = StationaryTrackFilter(
        movement_px=args.ball_stationary_px,
        static_frames=max(1, args.ball_stationary_frames),
    )
    moving_ball_filter = MovingBallFilter(
        history_frames=max(2, args.ball_motion_history),
        min_travel_px=args.ball_min_travel,
    )
    ball_ai_assist = None
    if args.ball_ai_assist:
        ball_ai_assist = OnlineBallTrackAssist(
            learning_rate=max(0.0, args.ball_ai_learning_rate),
            influence=max(0.0, args.ball_ai_influence),
            update=not args.no_ball_ai_online_learning,
            model_path=args.ball_ai_model,
        )
    ball_selector = BallSelector(
        seed_confirm_frames=max(1, args.ball_seed_frames),
        seed_min_travel_px=max(0.0, args.ball_seed_min_travel),
        seed_match_distance=max(1.0, args.ball_seed_match_distance),
        pending_miss_tolerance=max(0, args.ball_seed_miss_tolerance),
        coast_frames=max(0, args.ball_coast_frames),
        max_jump_px=max(1.0, args.ball_max_jump_px),
        max_prediction_error_px=max(1.0, args.ball_max_prediction_error_px),
        max_side_flip_jump_px=max(1.0, args.ball_max_side_flip_jump_px),
        ai_assist=ball_ai_assist,
    )
    event_detector = EventDetector(
        bounce_min_vertical_change=args.bounce_min_vertical_change,
        bounce_min_gap_frames=max(1, args.bounce_min_gap_frames),
        bounce_x_margin_ratio=args.bounce_x_margin_ratio,
        bounce_y_margin_ratio=args.bounce_y_margin_ratio,
        player_hit_margin_px=max(0, args.player_hit_margin_px),
        racket_hit_margin_px=max(0, args.racket_hit_margin_px),
        min_event_travel=args.event_min_travel,
        player_hit_upper_body_ratio=args.player_hit_upper_body_ratio,
        hit_min_gap_frames=max(1, args.hit_min_gap_frames),
        hit_min_angle_change_deg=max(0.0, args.hit_min_angle_change_deg),
        hit_min_speed_change_ratio=max(0.0, args.hit_min_speed_change_ratio),
        bounce_min_y_ratio=max(0.0, min(1.0, args.bounce_min_y_ratio)),
        inv_court_homography=inv_court_homography,
        fallback_net_y=fallback_net_y,
        net_segment=net_segment,
    )
    scene_tracker = SimpleTracker(max_distance=args.object_max_distance)
    frame_index = max(0, args.start_frame)
    processed_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        processed_frames += 1
        scene_result = run_track(
            scene_model,
            frame,
            conf=args.scene_conf,
            imgsz=args.imgsz,
            device=args.device,
            tracker=args.tracker,
        )
        scene_raw_detections = extract_detections(scene_result)
        scene_detections = scene_tracker.update(scene_raw_detections)
        near_player, far_player, other_objects = split_players(scene_detections)
        racket_detections = [det for det in scene_detections if det["class_name"] == "tennis racket"]
        ball_model_candidates, ball_model_debug = detect_ball_candidates(frame, ball_model, args, return_debug=True)
        scene_ball_detections = scene_ball_candidates(scene_raw_detections)
        ball_candidates = list(ball_model_candidates)
        ball_candidates.extend(scene_ball_detections)
        deduped_ball_candidates = dedupe_ball_detections(ball_candidates)
        if args.filter_contact_ball_candidates:
            contact_filtered_ball_candidates = filter_ball_candidates_by_contact_zone(
                deduped_ball_candidates,
                near_player,
                racket_detections,
                args.player_hit_margin_px,
                args.racket_hit_margin_px,
            )
        else:
            contact_filtered_ball_candidates = deduped_ball_candidates
        court_ball_candidates = filter_ball_candidates_by_court(contact_filtered_ball_candidates, inv_court_homography)
        tracked_ball_detections = ball_tracker.update(court_ball_candidates)
        stationary_ball_detections = ball_filter.filter(tracked_ball_detections)
        moving_ball_detections = moving_ball_filter.filter(stationary_ball_detections)
        ball = ball_selector.select(moving_ball_detections)
        ball_debug = {
            "counts": {
                **ball_model_debug,
                "scene_ball_count": len(scene_ball_detections),
                "combined_count": len(ball_candidates),
                "deduped_count": len(deduped_ball_candidates),
                "contact_filtered_count": len(contact_filtered_ball_candidates),
                "court_count": len(court_ball_candidates),
                "tracked_count": len(tracked_ball_detections),
                "stationary_count": len(stationary_ball_detections),
                "moving_count": len(moving_ball_detections),
            },
            "moving_filter": moving_ball_filter.last_debug,
            "selector": ball_selector.last_debug,
        }
        event = event_detector.update(
            frame_index,
            ball,
            frame.shape,
            [far_player, near_player],
            racket_detections,
        )

        if far_player is not None:
            draw_box(frame, far_player, PLAYER_FAR_COLOR, f"Player Far {format_track_suffix(far_player)}".strip())
        if near_player is not None:
            draw_box(frame, near_player, PLAYER_NEAR_COLOR, f"Player Near {format_track_suffix(near_player)}".strip())
        if not args.hide_other_objects:
            for det in other_objects:
                label = f"{det['class_name']} {format_track_suffix(det)}".strip()
                draw_box(frame, det, GENERIC_COLOR, label)
        if ball is not None:
            draw_ball(frame, ball, ball_trail)
        else:
            ball_trail.appendleft(None)
        draw_events(frame, event_detector.bounces)
        mini_court_layout = draw_mini_court(
            frame,
            enabled=not args.no_court_overlay,
            size=args.court_overlay_size,
            margin=args.court_overlay_margin,
        )
        ball_world_point = project_to_court_world(ball["center"], inv_court_homography) if ball is not None else None
        bounce_world_points = []
        for bounce in event_detector.bounces:
            world_point = bounce.get("world_point")
            if isinstance(world_point, list) and len(world_point) == 2:
                bounce_world_points.append((float(world_point[0]), float(world_point[1])))
        draw_mini_court_points(frame, mini_court_layout, ball_world_point, bounce_world_points)

        status = [
            f"frame={frame_index}",
            f"ball={'yes' if ball is not None else 'no'}",
            f"bounces={len(event_detector.bounces)}",
            f"players={len([p for p in (far_player, near_player) if p is not None])}",
            f"objects={len(scene_detections)}",
        ]
        cv2.putText(
            frame,
            "  ".join(status),
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if log_handle is not None:
            write_frame_log(
                log_handle,
                frame_index,
                ball,
                near_player,
                far_player,
                scene_detections,
                event,
                ball_debug=ball_debug,
            )
        if writer is not None:
            writer.write(frame)
        if not args.headless:
            cv2.imshow("YOLO Tennis Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
        if args.max_frames > 0 and processed_frames >= args.max_frames:
            break

    cleanup_outputs()


if __name__ == "__main__":
    main()

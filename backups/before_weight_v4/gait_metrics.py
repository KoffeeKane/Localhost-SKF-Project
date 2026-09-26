"""Measured lift events, separate from the oscillator's commanded swing phase."""
import numpy as np


class GaitMetrics:
    def __init__(self, info):
        self.max_height = {s: float(info.get(f"{s}_foot_height", 0)) for s in ("left", "right")}
        self.previous_rel = {s: float(info.get(f"{s}_foot_relative_x", 0)) for s in self.max_height}
        self.start_rel = dict(self.previous_rel)
        self.active_half = None
        self.events = []
        self.reward_sums = {}
        self.count = 0
        self.slip_sum = 0.0
        self.min_torso = 1.0

    def update(self, info):
        if "phase" not in info:
            return
        self.count += 1
        s = info["swing_leg"]
        if s != self.active_half:
            self.active_half = s
            self.start_rel[s] = self.previous_rel[s]
            self.event_recorded = False
        stance = "right" if s == "left" else "left"
        for side in self.max_height:
            self.max_height[side] = max(self.max_height[side], info[f"{side}_foot_height"])
            self.previous_rel[side] = info[f"{side}_foot_relative_x"]
            self.slip_sum += info[f"{side}_foot_slip_speed"]
        self.min_torso = min(self.min_torso, info["torso_upright"])
        # Require actual clearance, stance support, and 2 cm forward travel relative to pelvis.
        if (not self.event_recorded and info[f"{s}_foot_height"] > 0.02
                and not info[f"{s}_foot_contact"] and info[f"{stance}_foot_contact"]
                and info[f"{s}_foot_relative_x"] - self.start_rel[s] > 0.02
                and info["upright"] > 0.75 and info["torso_upright"] > 0.75):
            self.events.append(dict(leg=s, step=info["episode_step"], phase=info["phase"]))
            self.event_recorded = True
        for name, value in info["reward_terms"].items():
            self.reward_sums[name] = self.reward_sums.get(name, 0.0) + float(value)

    def summary(self):
        legs = [e["leg"] for e in self.events]
        alternating = any(a != b for a, b in zip(legs, legs[1:]))
        sustained = len(legs) >= 4 and all(a != b for a, b in zip(legs, legs[1:]))
        return dict(max_left_foot_height=self.max_height["left"], max_right_foot_height=self.max_height["right"],
                    alternating_foot_lift=alternating, sustained_alternating_steps=sustained,
                    qualified_lift_events=self.events, min_torso_upright=self.min_torso,
                    mean_contact_slip_speed=self.slip_sum / max(1, 2 * self.count),
                    mean_reward_terms={k: v / max(1, self.count) for k, v in self.reward_sums.items()})

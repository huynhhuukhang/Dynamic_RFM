"""Track segment identity across snapshots and detect evolution events."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import numpy as np
from .config import TrackingConfig


def jaccard(a: Set[int], b: Set[int]) -> float:
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class TrackedSegment:
    seg_id: str
    users: Set[int]
    centroid: np.ndarray
    size: int


@dataclass
class EvolutionRecord:
    snapshot: str
    event: str             # birth | death | split | merge | migration
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    users: Set[int] = field(default_factory=set)
    detail: Dict = field(default_factory=dict)


class SegmentTracker:
    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self._prev: Dict[str, TrackedSegment] = {}
        self._next_id = 0
        self._user_segment: Dict[int, str] = {}
        self.history: List[EvolutionRecord] = []

    def _new_id(self) -> str:
        sid = f"seg_{self._next_id}"
        self._next_id += 1
        return sid

    def update(self, snapshot_label: str,
               raw_labels: np.ndarray,
               user_ids: np.ndarray,
               centroids_raw: np.ndarray) -> Dict[int, str]:
        """
        raw_labels: cluster label per user (0..k-1)
        user_ids:   aligned user IDs
        centroids_raw: k x d raw-RFM centroids
        Returns mapping from raw label -> segment_id.
        """
        current: Dict[str, TrackedSegment] = {}
        new_label_to_id: Dict[int, str] = {}

        # Build per-cluster user sets
        per_cluster: Dict[int, Set[int]] = {}
        for lab, uid in zip(raw_labels, user_ids):
            per_cluster.setdefault(int(lab), set()).add(int(uid))

        # --- Matching phase: match new clusters to previous segments ---
        pairs: List[Tuple[float, int, str]] = []
        for lab, users in per_cluster.items():
            c = centroids_raw[lab]
            for sid, seg in self._prev.items():
                sim = cosine(c, seg.centroid) * 0.5 + jaccard(users, seg.users) * 0.5
                pairs.append((sim, lab, sid))
        pairs.sort(reverse=True, key=lambda x: x[0])

        used_prev: Set[str] = set()
        used_labels: Set[int] = set()
        for sim, lab, sid in pairs:
            if lab in used_labels or sid in used_prev:
                continue
            if sim < self.cfg.match_similarity_threshold:
                continue
            new_label_to_id[lab] = sid
            used_prev.add(sid)
            used_labels.add(lab)

        # --- Remaining new labels: possible splits of an unmatched previous segment ---
        unmatched_labels = [lab for lab in per_cluster if lab not in new_label_to_id]
        unmatched_prev = [sid for sid in self._prev if sid not in used_prev]

        # Try to detect splits: multiple new labels with high overlap to one prev segment
        for sid in unmatched_prev:
            parent_users = self._prev[sid].users
            overlapping: List[Tuple[float, int]] = []
            for lab in list(unmatched_labels):
                ov = len(parent_users & per_cluster[lab]) / max(1, len(parent_users))
                if ov > 0.3:
                    overlapping.append((ov, lab))
            if len(overlapping) >= self.cfg.split_min_children:
                child_ids = []
                for _, lab in overlapping:
                    cid = self._new_id()
                    new_label_to_id[lab] = cid
                    child_ids.append(cid)
                    used_labels.add(lab)
                    unmatched_labels.remove(lab)
                self.history.append(EvolutionRecord(
                    snapshot=snapshot_label, event="split",
                    parent_ids=[sid], child_ids=child_ids,
                ))
                used_prev.add(sid)

        # --- Remaining new labels: births ---
        for lab in list(unmatched_labels):
            cid = self._new_id()
            new_label_to_id[lab] = cid
            used_labels.add(lab)
            self.history.append(EvolutionRecord(
                snapshot=snapshot_label, event="birth", child_ids=[cid],
                users=per_cluster[lab],
            ))

        # --- Remaining prev segments: deaths or merges ---
        # Merge detection: a prev segment whose users mostly land in one new label already covered.
        remaining_prev = [sid for sid in self._prev if sid not in used_prev]
        for sid in remaining_prev:
            parent_users = self._prev[sid].users
            if not parent_users:
                continue
            # Which new labels absorbed most of the parent's users?
            absorptions: Dict[int, float] = {}
            for lab, users in per_cluster.items():
                ov = len(parent_users & users) / len(parent_users)
                if ov > 0.1:
                    absorptions[lab] = ov
            if not absorptions:
                self.history.append(EvolutionRecord(
                    snapshot=snapshot_label, event="death", parent_ids=[sid],
                ))
                continue
            top_labs = sorted(absorptions.items(), key=lambda kv: kv[1], reverse=True)
            # Merge if one prev segment's users were split into 2+ existing/new clusters
            if len(top_labs) >= 2:
                self.history.append(EvolutionRecord(
                    snapshot=snapshot_label, event="merge",
                    parent_ids=[sid],
                    child_ids=[new_label_to_id[lab] for lab, _ in top_labs if lab in new_label_to_id],
                ))
            # else: absorbed into an existing segment -> no death but user migration only

        # --- Build tracked segments ---
        for lab, sid in new_label_to_id.items():
            current[sid] = TrackedSegment(
                seg_id=sid, users=set(per_cluster[lab]),
                centroid=centroids_raw[lab].copy(),
                size=len(per_cluster[lab]),
            )

        # --- Customer migration detection ---
        new_user_segment: Dict[int, str] = {}
        for lab, sid in new_label_to_id.items():
            for u in per_cluster[lab]:
                new_user_segment[u] = sid
        for uid, new_sid in new_user_segment.items():
            prev_sid = self._user_segment.get(uid)
            if prev_sid is not None and prev_sid != new_sid:
                self.history.append(EvolutionRecord(
                    snapshot=snapshot_label, event="migration",
                    parent_ids=[prev_sid], child_ids=[new_sid],
                    users={uid},
                ))

        self._prev = current
        self._user_segment = new_user_segment
        return new_label_to_id
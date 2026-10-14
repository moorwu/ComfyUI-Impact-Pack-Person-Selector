"""Pure logic for Person Detailer: no torch / comfy imports (unit-testable without a GPU)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import numpy as np

_INDEX_HELP = "valid index examples: '2', '1,3', '2-4', '-1' (last person)"
_SINGLE_RE = re.compile(r"^-?\d+$")
_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")


def parse_index_expr(expr: str, count: int) -> tuple[list[int], list[str]]:
    expr = expr.strip()
    if not expr:
        return list(range(1, count + 1)), []

    ids: list[int] = []
    warnings: list[str] = []

    def add(token: str, value: int):
        if 1 <= value <= count:
            if value not in ids:
                ids.append(value)
        else:
            warnings.append(f"index {token} out of range (1..{count}), ignored")

    for raw in expr.split(","):
        token = raw.strip()
        if not token:
            raise ValueError(f"Person Selector: empty item in index '{expr}'; {_INDEX_HELP}")

        m = _RANGE_RE.match(token)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            if start < 1 or end < start:
                raise ValueError(f"Person Selector: bad index range '{token}'; {_INDEX_HELP}")
            for v in range(start, end + 1):
                add(str(v), v)
            continue

        if _SINGLE_RE.match(token):
            v = int(token)
            if v == 0:
                raise ValueError(f"Person Selector: index starts at 1, got '0'; {_INDEX_HELP}")
            add(token, count + 1 + v if v < 0 else v)
            continue

        raise ValueError(f"Person Selector: cannot parse index item '{token}'; {_INDEX_HELP}")

    return ids, warnings


_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _as_id(value) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid id {value!r}")  # noqa: TRY004 - ValueError intentional; person_vlm catches it to trigger the strict retry
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"invalid id {value!r}")


def parse_ids_json(text: str, valid_ids: set[int]) -> list[int]:
    for match in _OBJECT_RE.finditer(text):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "ids" not in obj:
            continue
        if not isinstance(obj["ids"], list):
            raise ValueError(f"'ids' is not a list in VLM output: {text!r}")  # noqa: TRY004 - ValueError intentional; person_vlm catches it to trigger the strict retry

        result: list[int] = []
        for value in obj["ids"]:
            i = _as_id(value)
            if i not in valid_ids:
                raise ValueError(f"id {i} not among candidates {sorted(valid_ids)}")
            if i not in result:
                result.append(i)
        return result

    raise ValueError(f"no JSON object with 'ids' found in VLM output: {text!r}")


# ---------------------------------------------------------------- geometry

Box = tuple[int, int, int, int]


@dataclass
class Detection:
    bbox: Box
    confidence: float


@dataclass
class PersonCandidate:
    person_box: Box
    confidence: float
    person_det: int | None = None
    face_box: Box | None = None
    face_det: int | None = None
    face_confidence: float | None = None
    index: int = 0
    area_ratio: float = 0.0
    sharpness: float | None = None
    excluded_reason: str | None = None
    head_box: Box | None = None


def box_area(b: Box) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _intersection_area(a: Box, b: Box) -> int:
    return box_area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _contains_point(b: Box, p: tuple[float, float]) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def synth_body_box(face: Box, image_w: int, image_h: int) -> Box:
    x1, y1, x2, y2 = face
    fw, fh = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    return (
        max(0, round(cx - 1.5 * fw)),
        max(0, round(y1 - 0.5 * fh)),
        min(image_w, round(cx + 1.5 * fw)),
        min(image_h, round(y1 + 7 * fh)),
    )


def _face_fits_person(face: Box, person: Box) -> bool:
    fx, fy = _center(face)
    px1, py1, px2, py2 = person
    return px1 <= fx <= px2 and py1 <= fy <= py1 + 0.5 * (py2 - py1)


def face_short_side(box: Box | None) -> int:
    return 0 if box is None else max(0, min(box[2] - box[0], box[3] - box[1]))


def relative_scale(c: PersonCandidate, largest_face: int, largest_body: int, largest_head: int = 0) -> float:
    if largest_head > 0 and c.head_box is not None:
        return face_short_side(c.head_box) / largest_head
    if c.face_box is not None and largest_face > 0:
        return face_short_side(c.face_box) / largest_face
    if largest_body > 0:
        return math.sqrt(box_area(c.person_box) / largest_body)
    return 1.0


def anchor_box(c: PersonCandidate) -> Box | None:
    return c.face_box if c.face_box is not None else c.head_box


def anchor(c: PersonCandidate) -> tuple[float, float]:
    box = anchor_box(c)
    return _center(box) if box is not None else _center(c.person_box)


def associate(persons: list[Detection], faces: list[Detection], image_w: int, image_h: int,
              person_masks: list[np.ndarray] | None = None) -> list[PersonCandidate]:
    if person_masks is not None and (len(person_masks) != len(persons)
                                    or any(np.asarray(m).shape != (image_h, image_w) for m in person_masks)):
        raise ValueError("associate: person_masks must contain one full-image mask per person")
    cands = [PersonCandidate(person_box=p.bbox, confidence=p.confidence, person_det=i) for i, p in enumerate(persons)]

    def mask_rank(pi: int, face: Box) -> tuple[float, float]:
        if person_masks is None:
            return 0.0, 0.0
        mask = np.asarray(person_masks[pi])
        cx, cy = _center(face)
        x1, y1 = max(0, face[0]), max(0, face[1])
        x2, y2 = min(image_w, face[2]), min(image_h, face[3])
        region = mask[y1:y2, x1:x2] > 0.5
        centre_inside = mask[min(image_h - 1, max(0, int(cy))), min(image_w - 1, max(0, int(cx)))] > 0.5
        return -float(centre_inside), -float(region.mean()) if region.size else 0.0

    claimed: dict[int, list[int]] = {}
    orphans: list[int] = []
    for fi, face in enumerate(faces):
        eligible = [pi for pi, p in enumerate(persons) if _face_fits_person(face.bbox, p.bbox)]
        if not eligible:
            orphans.append(fi)
            continue
        face_area = max(box_area(face.bbox), 1)
        best = min(
            eligible,
            key=lambda pi: (*mask_rank(pi, face.bbox),
                            -_intersection_area(face.bbox, persons[pi].bbox) / face_area, box_area(persons[pi].bbox)),
        )
        claimed.setdefault(best, []).append(fi)

    for pi, face_ids in claimed.items():
        face_ids = sorted(face_ids, key=lambda fi: (_center(faces[fi].bbox)[1], -faces[fi].confidence))
        chosen = face_ids[0]
        cands[pi].face_box = faces[chosen].bbox
        cands[pi].face_det = chosen
        cands[pi].face_confidence = faces[chosen].confidence
        orphans.extend(face_ids[1:])

    for fi in sorted(orphans):
        face = faces[fi]
        cands.append(PersonCandidate(
            person_box=synth_body_box(face.bbox, image_w, image_h),
            confidence=face.confidence,
            face_box=face.bbox,
            face_det=fi,
            face_confidence=face.confidence,
        ))

    total = max(image_w * image_h, 1)
    for c in cands:
        c.area_ratio = box_area(c.person_box) / total
    return cands


def laplacian_variance(gray: np.ndarray) -> float:
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())


def apply_exclusions(cands: list[PersonCandidate], min_person_ratio: float, min_face_size: int, min_sharpness: float,
                     min_relative_size: float = 0.0) -> None:
    largest_face = max((face_short_side(c.face_box) for c in cands if c.face_box is not None), default=0)
    largest_body = max((box_area(c.person_box) for c in cands), default=0)
    largest_head = max((face_short_side(c.head_box) for c in cands if c.head_box is not None), default=0)
    for c in cands:
        c.excluded_reason = None
        if c.area_ratio < min_person_ratio:
            c.excluded_reason = "small"
        elif min_relative_size > 0 and relative_scale(c, largest_face, largest_body, largest_head) < min_relative_size:
            c.excluded_reason = "background"
        elif c.face_box is not None and face_short_side(c.face_box) < min_face_size:
            c.excluded_reason = "tiny_face"
        elif min_sharpness > 0 and c.sharpness is not None and c.sharpness < min_sharpness:
            c.excluded_reason = "blurry"


SORT_KEYS = ("left_to_right", "top_to_bottom", "area", "confidence")


def number_candidates(cands: list[PersonCandidate], sort_by: str) -> list[PersonCandidate]:
    keys = {
        "left_to_right": lambda c: (anchor(c)[0], anchor(c)[1]),
        "top_to_bottom": lambda c: (anchor(c)[1], anchor(c)[0]),
        "area": lambda c: -box_area(c.person_box),
        "confidence": lambda c: -c.confidence,
    }
    if sort_by not in keys:
        raise ValueError(f"Person Detector: unknown sort_by '{sort_by}', expected one of {SORT_KEYS}")

    for c in cands:
        c.index = 0
    included = sorted((c for c in cands if c.excluded_reason is None), key=keys[sort_by])
    for i, c in enumerate(included, start=1):
        c.index = i
    return included


# ---------------------------------------------------------------- masks (v2)


def expand_box(b: Box, frac: float, w: int, h: int) -> Box:
    dx = round((b[2] - b[0]) * frac)
    dy = round((b[3] - b[1]) * frac)
    return max(0, b[0] - dx), max(0, b[1] - dy), min(w, b[2] + dx), min(h, b[3] + dy)


def _box_region(h: int, w: int, box: Box | None) -> np.ndarray:
    region = np.zeros((h, w), dtype=bool)
    if box is not None:
        region[box[1]:box[3], box[0]:box[2]] = True
    return region


def exclusive_masks(masks: list[np.ndarray], face_boxes: list[Box | None], synthetic: list[bool],
                    face_margin: float = 0.15, preserve_contours: bool = False) -> list[np.ndarray]:
    """Disjoint full-image masks. Protect anchors, prefer detected bodies over synthetic
    rectangles, then give shared pixels to the nearest face/head (mask centre if absent).
    This resolves ownership only; silhouettes still require an instance detector or SAM.
    """
    if not len(masks) == len(face_boxes) == len(synthetic):
        raise ValueError(f"exclusive_masks: masks, face_boxes and synthetic must have the same length "
                         f"(got {len(masks)}, {len(face_boxes)}, {len(synthetic)})")
    if not masks:
        return []
    h, w = masks[0].shape
    if any(np.asarray(m).shape != (h, w) for m in masks):
        raise ValueError("exclusive_masks: all masks must have the same (H, W) shape")
    own_faces = [_box_region(h, w, fb) for fb in face_boxes]
    expanded_faces = [_box_region(h, w, None if fb is None else expand_box(fb, face_margin, w, h)) for fb in face_boxes]
    bodies = [np.asarray(m).astype(bool) for m in masks]

    out = []
    for i, body in enumerate(bodies):
        m = body.copy()
        for j in range(len(bodies)):
            if j == i:
                continue
            foreign_face = expanded_faces[j]
            if preserve_contours and not synthetic[i] and not synthetic[j]:
                # A neighbour's face box can extend into this person's valid hair/arm.
                # For silhouettes, carve only pixels actually claimed by both masks.
                foreign_face = foreign_face & bodies[j]
            m &= ~(foreign_face & ~own_faces[i])
            if synthetic[i] and not synthetic[j]:
                m &= ~(bodies[j] & ~own_faces[i])
        out.append(m)

    counts = np.zeros((h, w), dtype=np.int32)
    for m in out:
        counts += m
    ys, xs = np.nonzero(counts > 1)
    if not len(xs):
        return out

    centres = []
    for i, body in enumerate(bodies):
        if face_boxes[i] is not None:
            centres.append(_center(face_boxes[i]))
        else:
            by, bx = np.nonzero(body)
            centres.append(((float(bx.min()) + float(bx.max())) / 2,
                            (float(by.min()) + float(by.max())) / 2) if len(bx) else (0.0, 0.0))

    # Work only on shared pixels, without allocating an N x H x W distance tensor.
    owners = np.full(len(xs), -1, dtype=np.int32)
    best_priority = np.full(len(xs), 3, dtype=np.int8)
    best_distance = np.full(len(xs), np.inf)
    order = sorted(range(len(out)), key=lambda i: (centres[i], synthetic[i], int(bodies[i].sum())))
    for i in order:
        cx, cy = centres[i]
        distance = (xs - cx) ** 2 + (ys - cy) ** 2
        priority = np.where(own_faces[i][ys, xs], 0, 2 if synthetic[i] else 1)
        take = out[i][ys, xs] & ((priority < best_priority)
                                | ((priority == best_priority) & (distance < best_distance)))
        owners[take] = i
        best_priority[take] = priority[take]
        best_distance[take] = distance[take]
    for i, m in enumerate(out):
        m[ys, xs] = owners == i
    return out


def others_mask(masks: list[np.ndarray], i: int) -> np.ndarray:
    own = np.asarray(masks[i]).astype(bool)
    other = np.zeros(own.shape, dtype=bool)
    for j, m in enumerate(masks):
        if j != i:
            other |= np.asarray(m).astype(bool)
    return other & ~own


def sam_prompts(target_body: Box, target_face: Box | None, other_faces: list[Box],
                target_head: Box | None = None) -> tuple[list[tuple[float, float]], list[int], Box]:
    """other_faces: the other numbered people's anchor boxes (face, or head when the face isn't visible)."""
    own = target_face if target_face is not None else target_head
    points = [_center(own) if own is not None else _center(target_body)]
    labels = [1]
    for face in other_faces:
        cx, cy = _center(face)
        if (cx, cy) == points[0] or (own is not None and _contains_point(own, (cx, cy))):
            continue  # a negative inside the target's own face/head would fight the positive point
        # A neighbour's head may lie outside this box while their torso/arm overlaps it.
        # Keep that negative too, so a merged-person SAM alternative can be rejected.
        points.append((cx, cy))
        labels.append(0)
    return points, labels, target_body


def select_sam_mask(masks, points: list[tuple[float, float]], labels: list[int],
                    box: Box, image_shape: tuple[int, int]) -> np.ndarray | None:
    """SAM multimasks are alternatives, not separate parts of a person. Pick a single
    mask containing positive prompts and excluding negative prompts; reject merged people.
    Prefer the largest valid candidate to avoid choosing just the face over the body.
    """
    h, w = image_shape
    region = _box_region(h, w, box)
    samples = [(min(w - 1, max(0, int(x))), min(h - 1, max(0, int(y))), label)
               for (x, y), label in zip(points, labels)]
    best, best_area = None, 0
    for candidate in masks:
        m = np.asarray(candidate) > 0.5
        if m.shape != (h, w):
            raise ValueError(f"Person Detector: SAM mask shape {m.shape} does not match image {(h, w)}")
        if any(bool(m[y, x]) != bool(label) for x, y, label in samples):
            continue
        m = m & region
        area = int(m.sum())
        if area > best_area:
            best, best_area = m, area
    return best


# ---------------------------------------------------------------- multi-detector merge & heads (v2, F7)


def box_iou(a: Box, b: Box) -> float:
    inter = _intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(a: Box, b: Box) -> float:
    smaller = min(box_area(a), box_area(b))
    return _intersection_area(a, b) / smaller if smaller > 0 else 0.0


def _same_person(a: Box, b: Box, iou_thr: float, contain_thr: float) -> bool:
    return box_iou(a, b) >= iou_thr or containment(a, b) >= contain_thr


def dedupe_detections(dets: list[Detection], iou_thr: float = 0.6, contain_thr: float = 0.8) -> list[int]:
    kept: list[int] = []
    for i in sorted(range(len(dets)), key=lambda i: -dets[i].confidence):
        if not any(_same_person(dets[i].bbox, dets[k].bbox, iou_thr, contain_thr) for k in kept):
            kept.append(i)
    return kept


def _claim_heads(boxes: list[Box], heads: list[Detection], box_ids, head_ids) -> dict[int, int]:
    """One head per box: each head in head_ids goes to its single best box in box_ids (head centre in the box's upper
    half, then the largest share of the head inside the box, then the smaller box); a box claimed by several heads
    keeps the top-most one (then the higher confidence). The other heads stay unassigned. Returns {box: head}."""
    box_ids = sorted(box_ids)
    claims: dict[int, list[int]] = {}
    for hi in sorted(head_ids):
        head = heads[hi].bbox
        head_area = max(box_area(head), 1)
        fits = [(-_intersection_area(head, boxes[bi]) / head_area, box_area(boxes[bi]), bi)
                for bi in box_ids if _face_fits_person(head, boxes[bi])]
        if fits:
            claims.setdefault(min(fits)[2], []).append(hi)
    return {bi: min((_center(heads[hi].bbox)[1], -heads[hi].confidence, hi) for hi in his)[2]
            for bi, his in claims.items()}


def merge_person_detections(primary: list[Detection], backfill: list[Detection], heads: list[Detection] | None,
                            iou_thr: float = 0.6, contain_thr: float = 0.8,
                            reasons: list[str] | None = None) -> list[tuple[str, int]]:
    """Primary boxes (deduplicated) first; a backfill box is added only if it is the same person as no kept box and,
    when heads are given, fits a head no kept box owns (each kept box owns one head, see _claim_heads; an added
    backfill box then owns the top-most of its free heads). One line per dropped backfill box goes to `reasons`."""
    merged = [("primary", i) for i in dedupe_detections(primary, iou_thr, contain_thr)]
    kept_boxes = [primary[i].bbox for _, i in merged]
    free_heads: set[int] = set()
    if heads is not None:
        owners = _claim_heads(kept_boxes, heads, range(len(kept_boxes)), range(len(heads)))
        free_heads = set(range(len(heads))) - set(owners.values())
    for j in sorted(range(len(backfill)), key=lambda j: -backfill[j].confidence):
        box = backfill[j].bbox
        same = next((k for k in kept_boxes if _same_person(box, k, iou_thr, contain_thr)), None)
        if same is not None:
            if reasons is not None:
                reasons.append(f"backfill #{j} {box} dropped: same person as kept box {same} "
                               f"(iou={box_iou(box, same):.2f}, containment={containment(box, same):.2f})")
            continue
        if heads is not None:
            claim = _claim_heads([box], heads, [0], free_heads)
            if not claim:
                if reasons is not None:
                    reasons.append(f"backfill #{j} {box} dropped: no unowned head in its upper half")
                continue
            free_heads.discard(claim[0])
        merged.append(("backfill", j))
        kept_boxes.append(box)
    return merged


def best_iou_match(box: Box, others: list[Box], iou_thr: float = 0.6) -> int | None:
    best, best_iou = None, iou_thr
    for k, other in enumerate(others):
        iou = box_iou(box, other)
        if iou >= best_iou:
            best, best_iou = k, iou
    return best


def assign_heads(cands: list[PersonCandidate], heads: list[Detection]) -> None:
    """A person with a face first gets the free head containing the face centre that covers the largest share of the
    face, capped at 0.8 so a stray oversized head can't outrank a tighter-fitting own head on coverage alone
    (then the smaller head, then the nearer centre); the remaining heads then go
    to the remaining faceless people by the one-head-per-box rule (_claim_heads) -- a person with a face but no head
    found in the step above stays headless rather than taking a neighbour's head. No head is given to two people."""
    owners: dict[int, int] = {}
    face_pairs = sorted(
        (-min(_intersection_area(head.bbox, c.face_box) / max(box_area(c.face_box), 1), 0.8), box_area(head.bbox),
         math.dist(_center(head.bbox), _center(c.face_box)), ci, hi)
        for ci, c in enumerate(cands) if c.face_box is not None
        for hi, head in enumerate(heads) if _contains_point(head.bbox, _center(c.face_box))
    )
    for *_rank, ci, hi in face_pairs:
        if ci not in owners and hi not in owners.values():
            owners[ci] = hi
    owners.update(_claim_heads([c.person_box for c in cands], heads,
                               [ci for ci in range(len(cands)) if ci not in owners and cands[ci].face_box is None],
                               [hi for hi in range(len(heads)) if hi not in owners.values()]))
    for ci, hi in owners.items():
        cands[ci].head_box = heads[hi].bbox

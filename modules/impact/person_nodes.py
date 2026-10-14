"""Person Detector / Person Selector: pick specific people in group images for Detailer (SEGS)."""

from __future__ import annotations

import logging
import sys
from collections import Counter
from dataclasses import dataclass

import impact.core as core  # noqa: PLR0402 - match impact/detectors.py's absolute-import style deliberately
import impact.utils as utils  # noqa: PLR0402
import numpy as np
import torch
from impact.core import SEG

from . import person_core as pc
from . import person_vlm

__all__ = ["PersonDetectorSEGS", "PersonSelector"]

logger = logging.getLogger(__name__)

_EXCLUDED_TAG = {"small": "S", "background": "BG", "tiny_face": "F", "blurry": "B"}
_GRAY = (128, 128, 128)
_TINT_ALPHA = 0.35
_SAM_THRESHOLD = 0.93


@dataclass
class Person:
    index: int
    person_seg: SEG
    face_seg: SEG | None
    person_bbox: pc.Box
    face_bbox: pc.Box | None
    area_ratio: float
    sharpness: float | None
    excluded_reason: str | None
    mask: np.ndarray | None = None  # final full-image bool body mask (numbered people only)
    head_bbox: pc.Box | None = None


def _rect_mask(h: int, w: int, box: pc.Box) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[box[1]:box[3], box[0]:box[2]] = True
    return mask


def _rect_seg(image: torch.Tensor, box: pc.Box, crop_factor: float, label: str, confidence: float) -> SEG:
    h, w = image.shape[1], image.shape[2]
    crop_region = utils.make_crop_region(w, h, box, crop_factor)
    x1, y1, x2, y2 = crop_region
    cropped_mask = np.array(_rect_mask(h, w, box)[y1:y2, x1:x2], dtype=np.float32)  # own copy, not a full-image view
    return SEG(utils.crop_image(image, crop_region), cropped_mask, confidence, crop_region, box, label, None)


def _full_mask(seg: SEG, h: int, w: int) -> np.ndarray:
    full = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = seg.crop_region
    cropped = seg.cropped_mask
    if isinstance(cropped, torch.Tensor):
        cropped = cropped.cpu().numpy()
    cropped = np.asarray(cropped)
    full[y1:y2, x1:x2] = cropped[: y2 - y1, : x2 - x1] > 0.5
    return full


def _with_mask(seg: SEG, mask: np.ndarray) -> SEG:
    x1, y1, x2, y2 = seg.crop_region
    return seg._replace(cropped_mask=np.array(mask[y1:y2, x1:x2], dtype=np.float32))  # own copy, not a full-image view


def _sam_object(sam_model):
    if isinstance(sam_model, core.SAM2Wrapper):
        return sam_model
    if hasattr(sam_model, "sam_wrapper"):
        return sam_model.sam_wrapper
    raise ValueError("Person Detector: invalid sam_model; connect a SAMLoader (Impact)")


def _sam_masks(sam_model, image: torch.Tensor, prompts) -> list[np.ndarray | None]:
    sam_obj = _sam_object(sam_model)
    img = np.clip(255.0 * image[0].cpu().numpy(), 0, 255).astype(np.uint8)
    sam_obj.prepare_device()
    try:
        results = []
        for points, labels, box in prompts:
            masks = sam_obj.predict(img, points, labels, list(box), _SAM_THRESHOLD)
            results.append(pc.select_sam_mask(masks, points, labels, box, img.shape[:2]))
        return results
    finally:
        sam_obj.release_device()


def _tinted(image: torch.Tensor, masks_colors) -> torch.Tensor:
    arr = image[0].detach().cpu().numpy().copy()
    for mask, color in masks_colors:
        rgb = np.asarray(color, dtype=np.float32) / 255.0
        arr[mask] = arr[mask] * (1.0 - _TINT_ALPHA) + rgb * _TINT_ALPHA
    return torch.from_numpy(arr).unsqueeze(0)


def _label_anchor(p: Person) -> pc.Box | None:
    return p.face_bbox if p.face_bbox is not None else p.head_bbox


class PersonDetectorSEGS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "person_detector": ("BBOX_DETECTOR", {"tooltip": "Body detector, e.g. UltralyticsDetectorProvider segm/person_yolov8m-seg.pt. Ignored when person_segm_detector is connected"}),
                "face_detector": ("BBOX_DETECTOR", {"tooltip": "Face detector, e.g. UltralyticsDetectorProvider bbox/face_yolov8m.pt"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_person_ratio": ("FLOAT", {"default": 0.015, "min": 0.0, "max": 1.0, "step": 0.001,
                                               "tooltip": "Exclude people whose body box covers less than this fraction of the image"}),
                "min_relative_size": ("FLOAT", {"default": 0.31, "min": 0.0, "max": 1.0, "step": 0.01,
                                                "tooltip": "Exclude background people whose relative size is below this: head short side / largest head short side when head_detector is connected, otherwise face short side / largest face short side, otherwise sqrt of body area / largest body area. 0 = off"}),
                "min_face_size": ("INT", {"default": 24, "min": 0, "max": 4096, "step": 1,
                                          "tooltip": "Exclude people whose face box short side is smaller than this (pixels)"}),
                "min_sharpness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 1.0,
                                            "tooltip": "Exclude people whose face Laplacian variance is below this (blurry background people). 0 = off"}),
                "sort_by": (list(pc.SORT_KEYS),),
                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "drop_size": ("INT", {"min": 1, "max": 8192, "step": 1, "default": 10}),
            },
            "optional": {
                "sam_model": ("SAM_MODEL", {"tooltip": "Refine each person's body mask with SAM, prompted with their face (or head when no face is visible) as the positive point and neighbours' faces (or heads when no face) as negative points"}),
                "person_segm_detector": ("SEGM_DETECTOR", {"tooltip": "Instance-segmentation body detector (UltralyticsDetectorProvider SEGM_DETECTOR output). Replaces person_detector and gives per-person silhouettes instead of rectangles"}),
                "extra_person_detector": ("BBOX_DETECTOR", {"tooltip": "Additional body detector used first (e.g. bbox/person_detect_v1.1_m.pt for anime). person_segm_detector / person_detector boxes only backfill people it missed"}),
                "head_detector": ("BBOX_DETECTOR", {"tooltip": "Head detector (e.g. bbox/head_detect_v2.0_s_yv11.pt). Heads anchor numbering, labels and SAM prompts when a face isn't visible; size filtering then compares head sizes"}),
            },
        }

    RETURN_TYPES = ("PERSONS", "IMAGE")
    RETURN_NAMES = ("persons", "preview")
    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Person"
    DESCRIPTION = "Detects people and their faces, excludes small/background/blurry ones, and numbers the rest (left to right by face by default)."

    def doit(self, image, person_detector, face_detector, threshold, min_person_ratio, min_relative_size, min_face_size,
             min_sharpness, sort_by, crop_factor, drop_size, sam_model=None, person_segm_detector=None,
             extra_person_detector=None, head_detector=None):
        if image.shape[0] > 1:
            logger.warning("[Impact Pack] Person Detector: batch input detected, only the first image is used")
            image = image[0:1]
        h, w = image.shape[1], image.shape[2]

        base_detector = person_segm_detector if person_segm_detector is not None else person_detector
        base_items = list(base_detector.detect(image, threshold, 0, crop_factor, drop_size)[1])
        face_items = list(face_detector.detect(image, threshold, 0, crop_factor, drop_size)[1])
        head_items = list(head_detector.detect(image, threshold, 0, crop_factor, drop_size)[1]) if head_detector is not None else []

        def to_det(seg):
            return pc.Detection(tuple(int(v) for v in seg.bbox), float(seg.confidence))

        if extra_person_detector is None:
            person_items = base_items
            person_masks = [_full_mask(s, h, w) for s in base_items]
        else:
            extra_items = list(extra_person_detector.detect(image, threshold, 0, crop_factor, drop_size)[1])
            base_boxes = [to_det(s).bbox for s in base_items]
            if head_detector is not None and not head_items:
                logger.warning("[Impact Pack] Person Detector: head_detector found no heads; head-gated backfill is disabled for this image")
            merge_reasons: list[str] = []
            order = pc.merge_person_detections([to_det(s) for s in extra_items], [to_det(s) for s in base_items],
                                               [to_det(s) for s in head_items] if head_detector is not None else None,
                                               reasons=merge_reasons)
            for line in merge_reasons:
                log = logger.debug if "dropped: same person" in line else logger.info
                log("[Impact Pack] Person Detector: %s", line)
            person_items, person_masks = [], []
            for source, i in order:
                if source == "primary":
                    seg = extra_items[i]
                    box = to_det(seg).bbox
                    match = pc.best_iou_match(box, base_boxes) if person_segm_detector is not None else None
                    mask = _rect_mask(h, w, box)
                    if match is not None:
                        borrowed = _full_mask(base_items[match], h, w) & mask
                        if borrowed.any():
                            mask = borrowed
                else:
                    seg = base_items[i]
                    mask = _full_mask(seg, h, w)
                person_items.append(seg)
                person_masks.append(mask)

        cands = pc.associate([to_det(s) for s in person_items], [to_det(s) for s in face_items], w, h,
                             person_masks=person_masks)
        pc.assign_heads(cands, [to_det(s) for s in head_items])

        gray = image[0].cpu().numpy().mean(axis=2) * 255.0
        for c in cands:
            if c.face_box is not None:
                x1, y1, x2, y2 = c.face_box
                c.sharpness = pc.laplacian_variance(gray[y1:y2, x1:x2])

        pc.apply_exclusions(cands, min_person_ratio, min_face_size, min_sharpness, min_relative_size)
        numbered = pc.number_candidates(cands, sort_by)
        excluded = [c for c in cands if c.index == 0]

        def body_seg(c):
            if c.person_det is not None:
                return person_items[c.person_det]._replace(label="person")
            return _rect_seg(image, c.person_box, crop_factor, "person", c.confidence)

        def face_seg(c):
            return face_items[c.face_det]._replace(label="face") if c.face_det is not None else None

        base_masks = [person_masks[c.person_det] if c.person_det is not None
                      else _rect_mask(h, w, c.person_box) for c in numbered]
        refined = [False] * len(numbered)
        if sam_model is not None and numbered:
            prompts = [pc.sam_prompts(c.person_box, c.face_box,
                                      [pc.anchor_box(o) for o in numbered if o is not c and pc.anchor_box(o) is not None],
                                      target_head=c.head_box)
                       for c in numbered]
            for k, sam_mask in enumerate(_sam_masks(sam_model, image, prompts)):
                if sam_mask is not None:
                    base_masks[k] = sam_mask
                    refined[k] = True
                else:
                    logger.warning("[Impact Pack] Person Detector: person #%d has no SAM mask satisfying its prompts; keeping the detector mask", numbered[k].index)
        synthetic = [c.person_det is None and not refined[k] for k, c in enumerate(numbered)]
        final_masks = pc.exclusive_masks(base_masks, [pc.anchor_box(c) for c in numbered], synthetic,
                                        preserve_contours=person_segm_detector is not None or sam_model is not None)

        persons = []
        for c, mask in zip(numbered, final_masks):
            if not mask.any():
                logger.warning("[Impact Pack] Person Detector: person #%d has no exclusive body pixels; its body mask is empty", c.index)
            persons.append(Person(c.index, _with_mask(body_seg(c), mask), face_seg(c), c.person_box, c.face_box,
                                  c.area_ratio, c.sharpness, None, mask, head_bbox=c.head_box))
        for c in excluded:
            persons.append(Person(0, body_seg(c), face_seg(c), c.person_box, c.face_box,
                                  c.area_ratio, c.sharpness, c.excluded_reason, head_bbox=c.head_box))

        tinted = _tinted(image, [(p.mask, person_vlm.palette(p.index - 1)) for p in persons if p.index > 0])
        boxes = []
        for p in persons:
            if p.index > 0:
                boxes.append((p.person_bbox, str(p.index), person_vlm.palette(p.index - 1), _label_anchor(p)))
            else:
                boxes.append((p.person_bbox, "x" + _EXCLUDED_TAG.get(p.excluded_reason, "?"), _GRAY))
        preview = person_vlm.draw_boxes(tinted, boxes)

        return {"image_size": (h, w), "persons": persons}, preview


_SELECTED = (0, 200, 0)
_REMAINED = (200, 200, 200)


def _segs(size, people: list[Person], attr: str):
    return size, [getattr(p, attr) for p in people if getattr(p, attr) is not None]


class PersonSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "persons": ("PERSONS",),
                "image": ("IMAGE",),
                "index": ("STRING", {"default": "", "tooltip": "Person numbers from Person Detector: '2', '1,3', '2-4', '-1' (last). Empty = everyone"}),
                "gender": (["any", "male", "female"],),
                "description": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "Character name or natural-language description, e.g. 'short hair, red jacket'. Requires clip"}),
                "verify": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled",
                                       "tooltip": "Re-check each picked person on a crop (other people greyed out) and keep those with P(yes) >= verify_threshold"}),
                "verify_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": sys.maxsize}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Krea 2 text encoder (CLIPLoader type 'krea2'); needed for gender/description"}),
            },
        }

    RETURN_TYPES = ("SEGS", "SEGS", "SEGS", "SEGS", "IMAGE", "STRING")
    RETURN_NAMES = ("face_SEGS", "person_SEGS", "remained_face_SEGS", "remained_person_SEGS", "preview", "debug_text")
    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Person"
    DESCRIPTION = "Selects people by number, gender, character name or description (Qwen3-VL via the Krea 2 CLIP)."

    def doit(self, persons, image, index, gender, description, verify, verify_threshold, seed, clip=None):
        if image.shape[0] > 1:
            logger.warning("[Impact Pack] Person Selector: batch input detected, only the first image is used")
        image = image[0:1]
        size = persons["image_size"]
        image_size = (image.shape[1], image.shape[2])
        if image_size != tuple(size):
            raise ValueError(f"Person Selector: image size {image_size} does not match the Person Detector's "
                             f"image size {tuple(size)}; connect the same image to both nodes.")
        numbered = [p for p in persons["persons"] if p.index > 0]
        excluded = [p for p in persons["persons"] if p.index == 0]

        log: list[str] = []
        ids, warnings = pc.parse_index_expr(index, len(numbered))
        log.extend(warnings)
        chosen = [numbered[i - 1] for i in ids]

        if gender != "any" or description.strip():
            condition = person_vlm.build_condition(gender, description)
            picked, dropped, raw = person_vlm.pick_ids(clip, image, [(p.index, p.person_bbox, _label_anchor(p)) for p in numbered],
                                                       [p.index for p in chosen], condition, seed=seed)
            log.append(f"condition: {condition}")
            log.append(f"vlm answer: {raw}")
            if dropped:
                log.append(f"ignored non-candidate ids from VLM: {dropped}")
            chosen = [p for p in chosen if p.index in picked]

            if verify:
                masks = [p.mask for p in numbered]
                have_masks = all(m is not None for m in masks)
                kept = []
                for p in chosen:
                    x1, y1, x2, y2 = p.person_bbox
                    if x2 <= x1 or y2 <= y1:
                        log.append(f"verify #{p.index}: skipped (empty crop)")
                        continue
                    crop = image[:, y1:y2, x1:x2, :].clone()
                    if have_masks:
                        k = next(i for i, q in enumerate(numbered) if q is p)
                        other = pc.others_mask(masks, k)[y1:y2, x1:x2]
                        crop[0][torch.from_numpy(other)] = 0.5
                    prob = person_vlm.yes_probability(clip, crop, condition)
                    log.append(f"verify #{p.index}: P(yes)={prob:.3f}")
                    if prob >= verify_threshold:
                        kept.append(p)
                chosen = kept
        elif verify:
            log.append("verify ignored: no gender/description condition")

        chosen = sorted(chosen, key=lambda p: p.index)
        chosen_ids = {p.index for p in chosen}
        remained = [p for p in numbered if p.index not in chosen_ids]

        reason_counts = Counter(str(p.excluded_reason) for p in excluded)
        reasons = ", ".join(f"{reason} x{count}" if count > 1 else reason for reason, count in reason_counts.items())
        header = [
            "selected=" + ",".join(str(p.index) for p in chosen),
            f"numbered={len(numbered)} excluded={len(excluded)}" + (f" ({reasons})" if reasons else ""),
        ]
        debug_text = "\n".join(header + log)

        boxes = [(p.person_bbox, str(p.index), _SELECTED if p.index in chosen_ids else _REMAINED, _label_anchor(p))
                 for p in numbered]
        preview = person_vlm.draw_boxes(image, boxes)

        result = (
            _segs(size, chosen, "face_seg"),
            _segs(size, chosen, "person_seg"),
            _segs(size, remained, "face_seg"),
            _segs(size, remained, "person_seg"),
            preview,
            debug_text,
        )
        return {"ui": {"text": [debug_text]}, "result": result}

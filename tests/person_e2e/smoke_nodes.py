"""Smoke test for person nodes with fake detectors (no model files, no GPU work).

Run ON THE SERVER from the synced repo root:
  COMFY_ROOT=/path/to/ComfyUI python3 tests/person_e2e/smoke_nodes.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMFY = Path(os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
sys.path[:0] = [str(COMFY), str(ROOT / "modules")]

import numpy as np
import torch
from impact import person_core as pc
from impact import person_nodes as pn
from impact import utils
from impact.core import SEG

H, W = 400, 800


class FakeDetector:
    def __init__(self, boxes, label):
        self.boxes = boxes
        self.label = label

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        items = []
        for box, conf in self.boxes:
            crop_region = utils.make_crop_region(W, H, box, crop_factor)
            mask = np.zeros((H, W), np.float32)
            mask[box[1]:box[3], box[0]:box[2]] = 1.0
            items.append(SEG(utils.crop_image(image, crop_region), utils.crop_ndarray2(mask, crop_region),
                             conf, crop_region, box, self.label, None))
        return (H, W), items


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok:", msg)


def make_image():
    rng = np.random.default_rng(0)
    img = rng.random((H, W, 3), dtype=np.float32)
    img[60:100, 330:370, :] = 0.5  # face of person C is perfectly flat -> blurry
    return torch.from_numpy(img).unsqueeze(0)


PERSON_BOXES = [((500, 50, 600, 390), 0.9), ((50, 50, 150, 390), 0.8), ((300, 50, 400, 390), 0.7), ((700, 380, 705, 385), 0.6)]
FACE_BOXES = [((530, 60, 570, 100), 0.9), ((80, 60, 120, 100), 0.9), ((330, 60, 370, 100), 0.9), ((700, 60, 740, 100), 0.5)]


def run_detector():
    image = make_image()
    persons, preview = pn.PersonDetectorSEGS().doit(
        image, FakeDetector(PERSON_BOXES, "person"), FakeDetector(FACE_BOXES, "face"),
        threshold=0.5, min_person_ratio=0.001, min_relative_size=0.25, min_face_size=16, min_sharpness=10.0,
        sort_by="left_to_right", crop_factor=3.0, drop_size=1)
    return image, persons, preview


def test_detector():
    _image, persons, preview = run_detector()
    items = persons["persons"]
    check(persons["image_size"] == (H, W), "image_size")
    included = [p for p in items if p.index > 0]
    excluded = {p.person_bbox: p.excluded_reason for p in items if p.index == 0}
    check([p.index for p in included] == [1, 2, 3], "three numbered people")
    check(included[0].person_bbox == (50, 50, 150, 390), "#1 is leftmost body")
    check(included[1].person_bbox == (500, 50, 600, 390), "#2 is body A")
    check(included[2].person_bbox == pc.synth_body_box((700, 60, 740, 100), W, H), "#3 is orphan face with synthetic body")
    check(excluded == {(300, 50, 400, 390): "blurry", (700, 380, 705, 385): "small"}, "blurry and small excluded")
    check(included[0].face_seg is not None and included[0].face_seg.label == "face", "face seg labelled")
    check(included[0].person_seg.label == "person", "person seg labelled")
    check(included[2].person_seg.bbox == included[2].person_bbox, "synthetic person seg bbox")
    check(float(included[2].person_seg.cropped_mask.max()) == 1.0, "synthetic person seg has rect mask")
    for p in included:
        cm = p.person_seg.cropped_mask
        x1, y1, x2, y2 = p.person_seg.crop_region
        check((cm.base is None or cm.base.size == cm.size) and cm.shape == (y2 - y1, x2 - x1),
              f"#{p.index} person seg mask is a compact crop-sized array (no full-image view)")
    check(tuple(preview.shape) == (1, H, W, 3), "preview shape")


def run_selector(persons, image, index="", gender="any", description="", verify=False, clip=None):
    out = pn.PersonSelector().doit(persons, image, index, gender, description, verify, 0.5, 0, clip=clip)
    return out["ui"], out["result"]


def test_selector():
    image, persons, _ = run_detector()

    ui, (face_segs, person_segs, rem_face, rem_person, preview, debug) = run_selector(persons, image, index="2")
    check(debug.splitlines()[0] == "selected=2", "index 2 selected")
    check(debug.splitlines()[1].startswith("numbered=3 excluded=2"), "stats line")
    check(ui["text"][0] == debug, "debug text exposed in ui")
    check([s.bbox for s in face_segs[1]] == [(530, 60, 570, 100)], "face SEGS of #2")
    check([s.bbox for s in person_segs[1]] == [(500, 50, 600, 390)], "person SEGS of #2")
    check(len(rem_face[1]) == 2 and len(rem_person[1]) == 2, "remained SEGS hold #1 and #3")
    check(face_segs[0] == (H, W), "SEGS carry image size")
    check(tuple(preview.shape) == (1, H, W, 3), "selector preview shape")

    _, (face_segs, person_segs, *_rest, debug) = run_selector(persons, image, index="")
    check(debug.splitlines()[0] == "selected=1,2,3", "empty index selects everyone")
    check(len(person_segs[1]) == 3, "three person SEGS")

    _, (*_segs, _preview, debug) = run_selector(persons, image, index="3,1")
    check(debug.splitlines()[0] == "selected=1,3", "output ordered by index")

    _, (face_segs, *_rest, debug) = run_selector(persons, image, index="9")
    check(debug.splitlines()[0] == "selected=" and "out of range" in debug, "out-of-range index warns")
    check(face_segs[1] == [], "nothing selected -> empty SEGS")

    _, (*_segs, _preview, debug) = run_selector(persons, image, index="1", verify=True)
    check("verify ignored" in debug, "verify without condition is ignored")

    try:
        run_selector(persons, image, gender="male")
    except Exception as e:  # noqa: BLE001 - plan-mandated smoke test, any exception is a failure to inspect
        check("krea2" in str(e), "gender without clip raises a clear error")
    else:
        raise AssertionError("expected error when gender is set without clip")

    bad_image = torch.zeros((1, H, W + 10, 3))
    try:
        run_selector(persons, bad_image, index="1")
    except ValueError as e:
        check("does not match" in str(e), "selector rejects image size mismatching the Detector's image")
    else:
        raise AssertionError("expected ValueError for mismatched image size")


class FakeSegmDetector:
    """Returns SEGs whose cropped_mask is an arbitrary silhouette (not the bbox)."""

    def __init__(self, items, label="person"):
        self.items = items  # list of (bbox, confidence, full_bool_mask)
        self.label = label

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        segs = []
        for box, conf, full in self.items:
            crop_region = utils.make_crop_region(W, H, box, crop_factor)
            segs.append(SEG(utils.crop_image(image, crop_region), utils.crop_ndarray2(full.astype(np.float32), crop_region),
                            conf, crop_region, box, self.label, None))
        return (H, W), segs


class ExplodingDetector:
    def detect(self, *args, **kwargs):
        raise AssertionError("bbox person_detector must not be called when person_segm_detector is connected")


def _mask(*rects):
    m = np.zeros((H, W), dtype=bool)
    for x1, y1, x2, y2 in rects:
        m[y1:y2, x1:x2] = True
    return m


def test_v2_segm_masks_exclusive_tint_and_grayout():
    image = torch.from_numpy(np.random.default_rng(1).random((H, W, 3), dtype=np.float32)).unsqueeze(0)
    p1_box, p2_box = (100, 50, 470, 390), (380, 50, 560, 390)
    p1_mask = _mask((140, 50, 260, 390), (260, 70, 470, 110))   # torso + an arm reaching over P2's face
    p2_mask = _mask((400, 50, 540, 390))
    segm = FakeSegmDetector([(p1_box, 0.9, p1_mask), (p2_box, 0.9, p2_mask)])
    faces = FakeDetector([((180, 60, 220, 100), 0.9), ((440, 60, 480, 100), 0.9)], "face")

    persons, preview = pn.PersonDetectorSEGS().doit(
        image, ExplodingDetector(), faces, threshold=0.5, min_person_ratio=0.0, min_relative_size=0.4,
        min_face_size=0, min_sharpness=0.0, sort_by="left_to_right", crop_factor=3.0, drop_size=1,
        person_segm_detector=segm)
    p1, p2 = persons["persons"][0], persons["persons"][1]
    check((p1.index, p2.index) == (1, 2) and p2.face_bbox == (440, 60, 480, 100), "segm detector used; faces associated")
    check(not p1.mask[60:100, 440:480].any(), "P1 mask no longer covers P2's face (F2)")
    check(bool(p1.mask[108, 300]), "P1 keeps arm pixels away from P2's face")
    check(float(np.asarray(p1.person_seg.cropped_mask).sum()) < (470 - 100) * (390 - 50), "person SEG mask is a silhouette, not the bbox")

    inside = (preview[0, 290:310, 190:210] - image[0, 290:310, 190:210]).abs().mean().item()
    outside = (preview[0, 290:310, 690:710] - image[0, 290:310, 690:710]).abs().mean().item()
    check(inside > 0.05 and outside < 0.01, "preview tints numbered masks only")

    captured = []
    orig_pick, orig_yes = pn.person_vlm.pick_ids, pn.person_vlm.yes_probability
    pn.person_vlm.pick_ids = lambda clip, img, people, candidate_ids, condition, seed=0: ([1], [], "fake")
    pn.person_vlm.yes_probability = lambda clip, crop, condition: captured.append(crop) or 1.0
    try:
        out = pn.PersonSelector().doit(persons, image, "", "any", "person one", True, 0.5, 0, clip=object())
    finally:
        pn.person_vlm.pick_ids, pn.person_vlm.yes_probability = orig_pick, orig_yes
    crop = captured[0]
    other = p2.mask[50:390, 100:470] & ~p1.mask[50:390, 100:470]
    check(other.any() and torch.allclose(crop[0][torch.from_numpy(other)], torch.tensor(0.5)), "verify crop greys out other people (F5)")
    check(out["result"][5].splitlines()[0] == "selected=1", "selector result with grey-out verify")


def test_v2b_extra_detector_merge_and_head_anchor():
    image = torch.from_numpy(np.random.default_rng(2).random((H, W, 3), dtype=np.float32)).unsqueeze(0)
    # anime-style primary detector: the hugging pair as two boxes, plus a fourth person; misses the far-right person
    extra = FakeDetector([((100, 50, 250, 390), 0.84), ((230, 50, 380, 390), 0.61), ((450, 50, 560, 390), 0.80)], "person")
    # realistic segm backfill: one merged hug box, the far-right person, and the fourth person's silhouette
    base = FakeSegmDetector([((100, 50, 380, 390), 0.9, _mask((110, 50, 370, 390))),
                             ((600, 50, 720, 390), 0.7, _mask((620, 60, 700, 390))),
                             ((455, 50, 565, 390), 0.85, _mask((470, 50, 550, 390)))])
    faces = FakeDetector([((200, 60, 240, 100), 0.9), ((235, 60, 275, 100), 0.9), ((490, 60, 530, 100), 0.9)], "face")
    heads = FakeDetector([((190, 50, 250, 110), 0.9), ((225, 50, 285, 110), 0.9), ((480, 50, 540, 110), 0.9),
                          ((630, 55, 690, 115), 0.9)], "head")

    persons, _preview = pn.PersonDetectorSEGS().doit(
        image, ExplodingDetector(), faces, threshold=0.5, min_person_ratio=0.0, min_relative_size=0.4,
        min_face_size=0, min_sharpness=0.0, sort_by="left_to_right", crop_factor=3.0, drop_size=1,
        person_segm_detector=base, extra_person_detector=extra, head_detector=heads)
    numbered = [p for p in persons["persons"] if p.index > 0]
    check([p.person_bbox for p in numbered] == [(100, 50, 250, 390), (230, 50, 380, 390), (450, 50, 560, 390), (600, 50, 720, 390)],
          "hug split by extra detector; missed person backfilled; merged box dropped")
    check(numbered[3].face_bbox is None and numbered[3].head_bbox == (630, 55, 690, 115), "back-turned person anchored by head")
    check(not numbered[2].mask[:, 450:470].any() and bool(numbered[2].mask[200, 500]), "primary box borrows segm silhouette (IoU>=0.6)")
    check(not numbered[0].mask[60:100, 241:250].any(), "person 1 mask excludes person 2's face (outside own face)")
    check(not numbered[1].mask[60:100, 230:235].any(), "person 2 mask excludes person 1's face (outside own face)")
    check(bool(numbered[3].mask[200, 650]) and not numbered[3].mask[200, 610], "backfilled person keeps its own silhouette")


def _detect(image, person_detector, faces, **kwargs):
    persons, _preview = pn.PersonDetectorSEGS().doit(
        image, person_detector, faces, threshold=0.5, min_person_ratio=0.0, min_relative_size=0.4,
        min_face_size=0, min_sharpness=0.0, sort_by="left_to_right", crop_factor=3.0, drop_size=1, **kwargs)
    return [p for p in persons["persons"] if p.index > 0]


def test_v2b_review_fixes():
    image = torch.from_numpy(np.random.default_rng(3).random((H, W, 3), dtype=np.float32)).unsqueeze(0)
    faces = FakeDetector([((150, 60, 190, 100), 0.9), ((640, 60, 680, 100), 0.9)], "face")

    # extra detector without head_detector: merge by overlap only
    extra = FakeDetector([((100, 50, 250, 390), 0.8)], "person")
    base = FakeDetector([((105, 50, 255, 390), 0.9), ((600, 50, 720, 390), 0.7)], "person")
    numbered = _detect(image, base, faces, extra_person_detector=extra)
    check([p.person_bbox for p in numbered] == [(100, 50, 250, 390), (600, 50, 720, 390)],
          "extra detector without head_detector: same-person base box dropped, distinct one backfilled")

    # bbox (non-segm) base detector: the extra box's mask stays its own rectangle (no silhouette borrowing)
    extra = FakeDetector([((450, 50, 560, 390), 0.8)], "person")
    base = FakeDetector([((455, 50, 565, 390), 0.9)], "person")
    numbered = _detect(image, base, FakeDetector([((490, 60, 530, 100), 0.9)], "face"), extra_person_detector=extra)
    check(len(numbered) == 1 and bool(numbered[0].mask[200, 452]) and numbered[0].mask.sum() == 110 * 340,
          "bbox base detector: extra box mask stays a rectangle")

    # faceless person anchored by a head: the neighbour's mask gives up that head region (F2 via head)
    p1_mask = _mask((140, 50, 260, 390), (260, 70, 470, 110))   # torso + an arm reaching over P2's head
    segm = FakeSegmDetector([((100, 50, 470, 390), 0.9, p1_mask), ((380, 50, 560, 390), 0.9, _mask((400, 50, 540, 390)))])
    heads = FakeDetector([((170, 50, 230, 110), 0.9), ((440, 55, 500, 115), 0.9)], "head")
    numbered = _detect(image, ExplodingDetector(), FakeDetector([((180, 60, 220, 100), 0.9)], "face"),
                       person_segm_detector=segm, head_detector=heads)
    p1, p2 = numbered
    check(p2.face_bbox is None and p2.head_bbox == (440, 55, 500, 115) and p1.head_bbox == (170, 50, 230, 110),
          "faceless person gets its own head; faced person gets the head around its face")
    check(not p1.mask[60:110, 440:500].any() and bool(p1.mask[108, 300]), "neighbour mask excludes faceless person's head (F2 via head)")


class FakeSAM:
    def __init__(self, masks):
        self.sam_wrapper = self
        self.masks = masks
        self.prepared = False
        self.released = False

    def prepare_device(self):
        self.prepared = True

    def release_device(self):
        self.released = True

    def predict(self, image, points, labels, box, threshold):
        return self.masks


def test_overlapping_bodies_and_sam_alternatives():
    image = make_image()
    boxes = [((100, 50, 350, 390), 0.9), ((250, 50, 500, 390), 0.9)]
    bodies = FakeDetector(boxes, "person")
    faces = FakeDetector([((150, 60, 190, 100), 0.9), ((410, 60, 450, 100), 0.9)], "face")
    numbered = _detect(image, bodies, faces)
    check(not (numbered[0].mask & numbered[1].mask).any(), "overlapping bbox bodies have no shared pixels")
    check(numbered[0].mask[300, 260] and numbered[1].mask[300, 340], "body overlap assigned to distinct people")

    own_a, own_b = _mask((140, 50, 300, 390)), _mask((300, 50, 460, 390))
    sam = FakeSAM([own_a | own_b, own_a, own_b])
    numbered = _detect(image, bodies, faces, sam_model=sam)
    check(sam.prepared and sam.released, "SAM device prepared and released")
    check(np.array_equal(numbered[0].mask, own_a) and np.array_equal(numbered[1].mask, own_b),
          "SAM merged-group candidate rejected; individual alternatives kept")
    for p in numbered:
        check(np.array_equal(pn._full_mask(p.person_seg, H, W), p.mask), "exported SEG matches exclusive preview mask")

    # Invalid SAM prompts fall back to the detector, which must still be exclusive.
    sam = FakeSAM([own_a | own_b])
    numbered = _detect(image, bodies, faces, sam_model=sam)
    check(not (numbered[0].mask & numbered[1].mask).any(), "SAM fallback does not restore body overlap")

    # A person with no identifiable pixels must not regain the original merged mask.
    merged = _mask((200, 100, 300, 390))
    segm = FakeSegmDetector([(box, conf, merged) for box, conf in boxes])
    numbered = _detect(image, ExplodingDetector(), FakeDetector([], "face"), person_segm_detector=segm)
    check(sum(int(p.mask.any()) for p in numbered) == 1, "empty exclusive body is kept empty")
    check(not (numbered[0].mask & numbered[1].mask).any(), "empty-body fallback never reintroduces merged people")

    sam = FakeSAM([np.ones((10, 10), dtype=bool)])
    try:
        pn._sam_masks(sam, image, [([(170, 80)], [1], boxes[0][0])])
    except ValueError:
        check(sam.released, "SAM device released after an invalid prediction")
    else:
        raise AssertionError("expected invalid SAM shape to raise")


if __name__ == "__main__":
    test_detector()
    test_selector()
    test_v2_segm_masks_exclusive_tint_and_grayout()
    test_v2b_extra_detector_merge_and_head_anchor()
    test_v2b_review_fixes()
    test_overlapping_bodies_and_sam_alternatives()
    print("SMOKE OK")

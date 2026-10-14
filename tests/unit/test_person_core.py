import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules"))

from impact import person_core as pc

# ---------- parse_index_expr ----------

def test_index_empty_selects_all():
    assert pc.parse_index_expr("  ", 3) == ([1, 2, 3], [])


def test_index_single_and_list():
    assert pc.parse_index_expr("2", 4) == ([2], [])
    assert pc.parse_index_expr("3, 1", 4) == ([3, 1], [])


def test_index_range_and_negative():
    assert pc.parse_index_expr("2-4", 5) == ([2, 3, 4], [])
    assert pc.parse_index_expr("-1", 5) == ([5], [])
    assert pc.parse_index_expr("-2,1", 5) == ([4, 1], [])


def test_index_dedupes_preserving_order():
    assert pc.parse_index_expr("2,1-3", 3) == ([2, 1, 3], [])


def test_index_out_of_range_warns_and_skips():
    ids, warnings = pc.parse_index_expr("1,7,-9", 3)
    assert ids == [1]
    assert len(warnings) == 2
    assert "7" in warnings[0] and "-9" in warnings[1]


@pytest.mark.parametrize("bad", ["1,,2", "a", "0", "3-1", "1-", "1.5", "--1"])
def test_index_syntax_errors(bad):
    with pytest.raises(ValueError, match="index"):
        pc.parse_index_expr(bad, 5)


# ---------- parse_ids_json ----------

def test_json_plain():
    assert pc.parse_ids_json('{"ids": [1, 3]}', {1, 2, 3}) == [1, 3]


def test_json_with_code_fence_and_chatter():
    text = 'Sure! Here you go:\n```json\n{"ids": [2]}\n```\nDone.'
    assert pc.parse_ids_json(text, {1, 2}) == [2]


def test_json_empty_list_is_valid():
    assert pc.parse_ids_json('{"ids": []}', {1, 2}) == []


def test_json_numeric_strings_and_dedupe():
    assert pc.parse_ids_json('{"ids": ["2", 2, 1]}', {1, 2}) == [2, 1]


def test_json_skips_objects_without_ids():
    assert pc.parse_ids_json('{"note": 1} {"ids": [1]}', {1}) == [1]


def test_json_out_of_candidates_raises():
    with pytest.raises(ValueError, match="not among candidates"):
        pc.parse_ids_json('{"ids": [4]}', {1, 2, 3})


@pytest.mark.parametrize("bad", ["no json here", '{"ids": "1"}', '{"ids": [true]}', '{"ids": [1.5]}'])
def test_json_invalid_raises(bad):
    with pytest.raises(ValueError):
        pc.parse_ids_json(bad, {1})


# ---------- geometry ----------

import numpy as np

D = pc.Detection


def test_synth_body_box_expands_and_clamps():
    # face 20x20 at (40,10); cx=50 -> x: 50-30..50+30, y: 10-10..10+140
    assert pc.synth_body_box((40, 10, 60, 30), 1000, 1000) == (20, 0, 80, 150)
    # clamped to image bounds
    assert pc.synth_body_box((0, 0, 20, 20), 50, 100) == (0, 0, 40, 100)


def test_associate_one_face_per_person():
    persons = [D((0, 0, 100, 300), 0.9), D((200, 0, 300, 300), 0.8)]
    faces = [D((220, 20, 260, 60), 0.7), D((30, 20, 70, 60), 0.6)]
    cands = pc.associate(persons, faces, 400, 300)
    assert len(cands) == 2
    assert cands[0].face_det == 1 and cands[0].face_box == (30, 20, 70, 60)
    assert cands[1].face_det == 0 and cands[1].face_confidence == 0.7
    assert cands[0].person_det == 0 and cands[0].confidence == 0.9
    assert cands[0].area_ratio == (100 * 300) / (400 * 300)


def test_face_in_lower_half_is_orphan():
    persons = [D((0, 0, 100, 200), 0.9)]
    faces = [D((30, 150, 50, 170), 0.5)]  # center y=160 > 0 + 0.5*200
    cands = pc.associate(persons, faces, 500, 500)
    assert cands[0].face_box is None
    assert len(cands) == 2
    assert cands[1].person_det is None
    assert cands[1].face_det == 0
    assert cands[1].confidence == 0.5
    assert cands[1].person_box == pc.synth_body_box((30, 150, 50, 170), 500, 500)


def test_face_goes_to_person_with_larger_overlap_ratio():
    persons = [D((0, 0, 100, 200), 0.9), D((50, 0, 300, 200), 0.9)]
    faces = [D((80, 20, 120, 50), 0.8)]  # center x=100: inside both; fully inside B only
    cands = pc.associate(persons, faces, 400, 400)
    assert cands[0].face_det is None
    assert cands[1].face_det == 0


def test_overlap_tie_goes_to_smaller_person():
    persons = [D((50, 0, 300, 200), 0.9), D((0, 0, 100, 200), 0.9)]
    faces = [D((60, 20, 90, 50), 0.8)]  # fully inside both
    cands = pc.associate(persons, faces, 400, 400)
    assert cands[1].face_det == 0
    assert cands[0].face_det is None


def test_multiple_faces_top_one_wins_rest_orphaned():
    persons = [D((0, 0, 200, 400), 0.9)]
    faces = [D((80, 100, 120, 140), 0.99), D((80, 20, 120, 60), 0.5)]
    cands = pc.associate(persons, faces, 400, 400)
    assert cands[0].face_det == 1  # higher on the image wins despite lower confidence
    assert len(cands) == 2 and cands[1].face_det == 0 and cands[1].person_det is None


def test_same_height_faces_tie_broken_by_confidence():
    persons = [D((0, 0, 200, 400), 0.9)]
    faces = [D((10, 20, 50, 60), 0.4), D((100, 20, 140, 60), 0.8)]
    cands = pc.associate(persons, faces, 400, 400)
    assert cands[0].face_det == 1


def test_laplacian_variance():
    assert pc.laplacian_variance(np.full((10, 10), 128.0)) == 0.0
    board = (np.indices((10, 10)).sum(axis=0) % 2) * 255.0
    assert pc.laplacian_variance(board) > 1000.0
    assert pc.laplacian_variance(np.zeros((2, 5))) == 0.0


def _cand(box, face=None, sharp=None, conf=0.5, ratio=0.1):
    return pc.PersonCandidate(person_box=box, confidence=conf, face_box=face, sharpness=sharp, area_ratio=ratio)


def test_exclusions():
    small = _cand((0, 0, 10, 10), ratio=0.001)
    tiny = _cand((0, 0, 100, 100), face=(0, 0, 10, 30))
    blurry = _cand((0, 0, 100, 100), face=(0, 0, 50, 50), sharp=5.0)
    no_face = _cand((0, 0, 100, 100), sharp=None)
    ok = _cand((0, 0, 100, 100), face=(0, 0, 50, 50), sharp=500.0)
    cands = [small, tiny, blurry, no_face, ok]
    pc.apply_exclusions(cands, min_person_ratio=0.01, min_face_size=16, min_sharpness=50.0)
    assert [c.excluded_reason for c in cands] == ["small", "tiny_face", "blurry", None, None]


def test_sharpness_zero_disables_blur_check():
    blurry = _cand((0, 0, 100, 100), face=(0, 0, 50, 50), sharp=0.1)
    blurry.excluded_reason = "blurry"  # stale value must be reset
    pc.apply_exclusions([blurry], min_person_ratio=0.0, min_face_size=0, min_sharpness=0.0)
    assert blurry.excluded_reason is None


def test_relative_size_linear_scale_without_faces():
    big = _cand((0, 0, 400, 800))       # area 320000 (largest)
    mid = _cand((500, 0, 700, 400))     # 80000 -> sqrt(0.25) = 0.5, kept at 0.4
    bg = _cand((800, 0, 900, 300))      # 30000 -> sqrt(0.094) = 0.306, background at 0.4
    pc.apply_exclusions([big, mid, bg], min_person_ratio=0.0, min_face_size=0, min_sharpness=0.0, min_relative_size=0.4)
    assert [c.excluded_reason for c in (big, mid, bg)] == [None, None, "background"]


def test_small_wins_over_background_and_zero_disables_relative():
    big = _cand((0, 0, 400, 800))
    tiny = _cand((0, 0, 10, 10), ratio=0.0001)
    pc.apply_exclusions([big, tiny], min_person_ratio=0.01, min_face_size=0, min_sharpness=0.0, min_relative_size=0.25)
    assert tiny.excluded_reason == "small"

    bg = _cand((800, 0, 900, 300))
    pc.apply_exclusions([big, bg], min_person_ratio=0.0, min_face_size=0, min_sharpness=0.0, min_relative_size=0.0)
    assert bg.excluded_reason is None


def test_number_left_to_right_skips_excluded():
    a = _cand((300, 0, 400, 100))
    b = _cand((0, 0, 100, 100))
    c = _cand((150, 0, 250, 100))
    c.excluded_reason = "small"
    ordered = pc.number_candidates([a, b, c], "left_to_right")
    assert ordered == [b, a]
    assert (b.index, a.index, c.index) == (1, 2, 0)


def test_number_other_sort_keys():
    big = _cand((0, 0, 300, 300), conf=0.3)
    low = _cand((0, 200, 50, 250), conf=0.9)
    assert pc.number_candidates([low, big], "area") == [big, low]
    assert pc.number_candidates([big, low], "confidence") == [low, big]
    assert pc.number_candidates([low, big], "top_to_bottom") == [big, low]


def test_number_bad_sort_key():
    with pytest.raises(ValueError, match="sort_by"):
        pc.number_candidates([], "random")


# ---------- edge cases ----------

def test_associate_empty_inputs():
    assert pc.associate([], [], 400, 300) == []


def test_associate_zero_persons_two_faces_are_orphans():
    faces = [D((10, 10, 50, 50), 0.9), D((100, 20, 140, 60), 0.8)]
    cands = pc.associate([], faces, 400, 300)
    assert len(cands) == 2
    for c, face in zip(cands, faces):
        assert c.person_det is None
        assert c.face_box == face.bbox
        assert c.person_box == pc.synth_body_box(face.bbox, 400, 300)


def test_associate_zero_size_face_box_does_not_crash():
    persons = [D((0, 0, 100, 200), 0.9)]
    faces = [D((10, 10, 10, 10), 0.5)]
    cands = pc.associate(persons, faces, 400, 300)
    assert len(cands) == 1
    assert cands[0].face_box == (10, 10, 10, 10)


def test_number_candidates_empty():
    assert pc.number_candidates([], "left_to_right") == []


# ---------- v2: F3 linear scale ----------

def test_relative_scale_uses_face_when_present():
    near = _cand((0, 0, 400, 800), face=(100, 50, 200, 150))       # face 100px, body 320000
    far_small_body = _cand((600, 0, 680, 200), face=(610, 10, 660, 60))  # face 50px -> 0.5; body sqrt(16000/320000)=0.22
    tiny_face = _cand((800, 0, 1000, 400), face=(850, 20, 890, 60))  # face 40px -> 0.4
    cands = [near, far_small_body, tiny_face]
    pc.apply_exclusions(cands, min_person_ratio=0.0, min_face_size=0, min_sharpness=0.0, min_relative_size=0.45)
    assert [c.excluded_reason for c in cands] == [None, None, "background"]


def test_relative_scale_values():
    assert pc.face_short_side(None) == 0
    assert pc.face_short_side((0, 0, 30, 50)) == 30
    c = _cand((0, 0, 100, 100), face=(0, 0, 20, 40))
    assert pc.relative_scale(c, largest_face=40, largest_body=10000) == 0.5
    faceless = _cand((0, 0, 50, 50))
    assert abs(pc.relative_scale(faceless, largest_face=40, largest_body=10000) - 0.5) < 1e-9


# ---------- v2: F4 face anchor ----------

def test_left_to_right_uses_face_center():
    fox = _cand((880, 0, 978, 600), face=(880, 100, 928, 150))      # body cx 929, face cx 904
    blonde = _cand((800, 0, 1064, 600), face=(825, 300, 873, 350))  # body cx 932, face cx 849
    ordered = pc.number_candidates([fox, blonde], "left_to_right")
    assert ordered == [blonde, fox]
    assert pc.anchor(fox) == (904.0, 125.0)
    assert pc.anchor(_cand((0, 0, 10, 20))) == (5.0, 10.0)


# ---------- v2: F2 exclusive masks ----------

def _rect(h, w, box):
    m = np.zeros((h, w), dtype=bool)
    m[box[1]:box[3], box[0]:box[2]] = True
    return m


def test_expand_box_clamps():
    assert pc.expand_box((10, 10, 30, 50), 0.15, 100, 100) == (7, 4, 33, 56)
    assert pc.expand_box((0, 0, 20, 20), 0.5, 25, 25) == (0, 0, 25, 25)


def test_exclusive_masks_remove_other_faces_but_keep_own_face():
    h, w = 100, 200
    a = _rect(h, w, (0, 0, 120, 100))                 # A's body box covers B's face
    b = _rect(h, w, (100, 0, 200, 100))
    face_a, face_b = (40, 10, 60, 30), (110, 10, 130, 30)
    out = pc.exclusive_masks([a, b], [face_a, face_b], [False, False])
    assert not out[0][10:30, 110:120].any()           # B's face removed from A
    assert out[0][10:30, 40:60].all()                 # A keeps its own face
    assert out[1][10:30, 110:130].all()               # B keeps its own face
    assert out[1][50:100, 100:120].all()              # shared body pixels are nearer B's face
    assert not (out[0] & out[1]).any()               # bodies, not just faces, must be exclusive


def test_exclusive_masks_synthetic_subtracts_other_bodies():
    h, w = 100, 200
    real = _rect(h, w, (0, 0, 100, 100))
    synth = _rect(h, w, (50, 0, 150, 100))
    out = pc.exclusive_masks([real, synth], [(20, 10, 40, 30), None], [False, True])
    assert not out[1][:, 50:100].any()
    assert out[1][:, 100:150].all()


def test_exclusive_masks_own_face_protected_from_neighbor_margin():
    h, w = 60, 100
    a = _rect(h, w, (0, 0, 50, 60))
    b = _rect(h, w, (50, 0, 100, 60))
    face_a, face_b = (30, 10, 49, 29), (50, 10, 69, 29)    # 1px apart; B's 15% margin (x 47..72) reaches into A's face box
    out = pc.exclusive_masks([a, b], [face_a, face_b], [False, False])
    assert out[0][10:29, 30:49].all()


def test_exclusive_masks_empty():
    assert pc.exclusive_masks([], [], []) == []


# ---------- v2: F5 others mask ----------

def test_others_mask_excludes_own_pixels():
    h, w = 10, 10
    a = _rect(h, w, (0, 0, 6, 10))
    b = _rect(h, w, (4, 0, 10, 10))
    other = pc.others_mask([a, b], 0)
    assert not other[:, 0:6].any()
    assert other[:, 6:10].all()


# ---------- v2: F6 SAM prompts ----------

def test_sam_prompts_include_neighbours_outside_body_box():
    points, labels, box = pc.sam_prompts((0, 0, 200, 400), (80, 20, 120, 60), [(150, 30, 190, 70), (300, 30, 340, 70)])
    assert points == [(100.0, 40.0), (170.0, 50.0), (320.0, 50.0)]
    assert labels == [1, 0, 0]
    assert box == (0, 0, 200, 400)


def test_sam_prompts_without_face_uses_body_center():
    points, labels, _ = pc.sam_prompts((0, 0, 100, 300), None, [])
    assert points == [(50.0, 150.0)] and labels == [1]


def test_sam_prompts_do_not_negate_faceless_positive_point():
    points, labels, _ = pc.sam_prompts((0, 0, 100, 300), None, [(40, 140, 60, 160)])
    assert points == [(50.0, 150.0)] and labels == [1]


# ---------- v2b: F7 merge, heads ----------

def test_box_iou_and_containment():
    a, b = (0, 0, 10, 10), (5, 0, 15, 10)
    assert abs(pc.box_iou(a, b) - 50 / 150) < 1e-9
    assert pc.containment((2, 2, 4, 4), (0, 0, 10, 10)) == 1.0
    assert pc.box_iou((0, 0, 0, 0), a) == 0.0
    assert pc.containment((0, 0, 0, 0), a) == 0.0


def test_dedupe_detections_keeps_higher_confidence():
    dets = [D((0, 0, 100, 200), 0.6), D((5, 5, 100, 200), 0.9), D((10, 20, 60, 120), 0.8), D((300, 0, 400, 200), 0.7)]
    assert pc.dedupe_detections(dets) == [1, 3]


def test_merge_primary_first_backfill_needs_new_head():
    primary = [D((0, 0, 100, 300), 0.8), D((120, 0, 220, 300), 0.7)]    # two hugging people, separate boxes
    backfill = [D((0, 0, 220, 300), 0.9),                                 # merged box containing both
                D((400, 0, 500, 300), 0.6),                               # person the primary detector missed
                D((600, 0, 700, 300), 0.5)]                               # box without any head
    heads = [D((30, 10, 70, 50), 0.9), D((150, 10, 190, 50), 0.9), D((430, 10, 470, 50), 0.9)]
    assert pc.merge_person_detections(primary, backfill, heads) == [("primary", 0), ("primary", 1), ("backfill", 1)]


def test_merge_without_heads_uses_overlap_only():
    primary = [D((0, 0, 100, 300), 0.8)]
    backfill = [D((10, 0, 110, 300), 0.9), D((400, 0, 500, 300), 0.6)]
    assert pc.merge_person_detections(primary, backfill, None) == [("primary", 0), ("backfill", 1)]


def test_best_iou_match():
    others = [(0, 0, 50, 100), (0, 0, 100, 100), (500, 0, 600, 100)]
    assert pc.best_iou_match((0, 0, 100, 100), others) == 1
    assert pc.best_iou_match((300, 0, 400, 100), others) is None


def test_assign_heads_and_anchor_prefers_face_then_head():
    back = _cand((0, 0, 100, 300))
    faced = _cand((200, 0, 300, 300), face=(230, 20, 270, 60))
    pc.assign_heads([back, faced], [D((20, 5, 80, 65), 0.9), D((225, 10, 275, 70), 0.9)])
    assert back.head_box == (20, 5, 80, 65) and faced.head_box == (225, 10, 275, 70)
    assert pc.anchor(back) == (50.0, 35.0)
    assert pc.anchor(faced) == (250.0, 40.0)
    assert pc.anchor_box(back) == (20, 5, 80, 65)
    assert pc.anchor_box(faced) == (230, 20, 270, 60)


def test_relative_scale_prefers_head_when_available():
    near = _cand((0, 0, 400, 800), face=(100, 50, 130, 80))      # turned face 30px
    far = _cand((600, 0, 680, 200), face=(610, 10, 670, 70))     # frontal face 60px (largest face)
    pc.apply_exclusions([near, far], min_person_ratio=0.0, min_face_size=0, min_sharpness=0.0, min_relative_size=0.6)
    assert near.excluded_reason == "background"                  # face-only scale 30/60 = 0.5
    near.head_box, far.head_box = (80, 20, 200, 140), (600, 0, 680, 80)   # heads 120px / 80px
    pc.apply_exclusions([near, far], min_person_ratio=0.0, min_face_size=0, min_sharpness=0.0, min_relative_size=0.6)
    assert near.excluded_reason is None and far.excluded_reason is None   # 1.0 and 0.667
    assert pc.relative_scale(far, largest_face=60, largest_body=320000, largest_head=120) == 80 / 120


def test_sam_prompts_uses_head_when_no_face():
    points, labels, _ = pc.sam_prompts((0, 0, 200, 400), None, [(120, 10, 180, 70)], target_head=(40, 0, 100, 60))
    assert points == [(70.0, 30.0), (150.0, 40.0)]
    assert labels == [1, 0]


def test_assign_heads_ignores_orphan_heads():
    person = _cand((0, 0, 100, 300))
    pc.assign_heads([person], [D((500, 0, 560, 60), 0.9)])
    assert person.head_box is None


# ---------- v2b review fixes: head ownership, face-consistent heads, merge reasons ----------

def test_merge_wide_box_owns_one_head_so_missed_neighbour_is_backfilled():
    primary = [D((0, 0, 300, 600), 0.9)]                  # A: wide box, its upper half also covers B's head
    backfill = [D((200, 0, 400, 600), 0.8)]               # B: person the extra detector missed (iou 0.25, containment 0.5)
    heads = [D((100, 10, 160, 70), 0.9), D((260, 30, 300, 70), 0.9)]
    assert pc.merge_person_detections(primary, backfill, heads) == [("primary", 0), ("backfill", 0)]


def test_merge_backfill_box_owns_only_one_new_head():
    primary = [D((0, 0, 100, 600), 0.9)]
    backfill = [D((300, 0, 500, 600), 0.8),               # wide box whose upper half holds two heads
                D((430, 0, 630, 600), 0.7)]               # the second head's own person (iou 0.21, containment 0.35)
    heads = [D((20, 10, 80, 70), 0.9), D((310, 10, 370, 70), 0.9), D((420, 40, 480, 100), 0.9)]
    assert pc.merge_person_detections(primary, backfill, heads) == [("primary", 0), ("backfill", 0), ("backfill", 1)]


def test_merge_empty_heads_disables_backfill_and_reports_each_box():
    primary = [D((0, 0, 100, 300), 0.8)]
    backfill = [D((400, 0, 500, 300), 0.6), D((600, 0, 700, 300), 0.7)]
    reasons: list[str] = []
    assert pc.merge_person_detections(primary, backfill, [], reasons=reasons) == [("primary", 0)]
    assert reasons == ["backfill #1 (600, 0, 700, 300) dropped: no unowned head in its upper half",
                       "backfill #0 (400, 0, 500, 300) dropped: no unowned head in its upper half"]


def test_merge_reasons_same_person_drop():
    primary = [D((0, 0, 100, 300), 0.8)]
    backfill = [D((10, 0, 110, 300), 0.9)]
    reasons: list[str] = []
    assert pc.merge_person_detections(primary, backfill, None, reasons=reasons) == [("primary", 0)]
    assert reasons == [("backfill #0 (10, 0, 110, 300) dropped: same person as kept box (0, 0, 100, 300) "
                        "(iou=0.82, containment=0.90)")]


def test_assign_heads_prefers_head_containing_own_face():
    faced = _cand((0, 0, 260, 600), face=(200, 60, 240, 100))
    neighbour = _cand((80, 0, 280, 600))                  # back-turned neighbour, box overlaps
    own_head, neighbour_head = D((190, 50, 250, 110), 0.9), D((60, 10, 120, 70), 0.9)   # neighbour's head is higher up
    pc.assign_heads([faced, neighbour], [own_head, neighbour_head])
    assert faced.head_box == own_head.bbox
    assert neighbour.head_box == neighbour_head.bbox


def test_assign_heads_face_head_not_given_twice():
    a = _cand((0, 0, 200, 600), face=(80, 40, 120, 80))
    b = _cand((0, 0, 210, 600), face=(90, 45, 130, 85))   # both face centres inside the one detected head
    pc.assign_heads([a, b], [D((60, 20, 150, 110), 0.9)])
    assert [a.head_box, b.head_box].count((60, 20, 150, 110)) == 1


def test_assign_heads_two_heads_in_one_box_assigns_only_top_most():
    person = _cand((0, 0, 200, 600))
    lower, top = D((120, 100, 180, 160), 0.95), D((20, 10, 80, 70), 0.6)
    pc.assign_heads([person], [lower, top])
    assert person.head_box == top.bbox


def test_assign_heads_face_without_own_head_does_not_take_neighbours_head():
    a = _cand((0, 0, 200, 600), face=(60, 60, 110, 110))          # A: face present, own head not detected
    b = _cand((100, 0, 400, 600))                                 # B: back-turned, no face
    b_head = D((140, 20, 200, 80), 0.9)                           # B's head only; not near A's face centre
    pc.assign_heads([a, b], [b_head])
    assert a.head_box is None
    assert b.head_box == b_head.bbox


def test_relative_scale_mixed_heads_people_without_head_fall_back_to_face_then_body():
    headed = _cand((0, 0, 400, 800), face=(10, 10, 60, 60))       # face 50px, head 100px
    headed.head_box = (0, 0, 100, 100)
    headless = _cand((450, 0, 550, 400), face=(460, 10, 500, 50))  # no head: face 40 / largest face 50 = 0.8
    bodyonly = _cand((600, 0, 700, 200))                           # no head, no face: sqrt(20000 / 320000) = 0.25
    assert pc.relative_scale(headless, largest_face=50, largest_body=320000, largest_head=100) == 0.8
    pc.apply_exclusions([headed, headless, bodyonly], min_person_ratio=0.0, min_face_size=0, min_sharpness=0.0,
                        min_relative_size=0.85)
    assert headed.excluded_reason is None
    assert headless.excluded_reason == "background"
    assert bodyonly.excluded_reason == "background"
    pc.apply_exclusions([headed, headless, bodyonly], min_person_ratio=0.0, min_face_size=0, min_sharpness=0.0,
                        min_relative_size=0.7)
    assert headless.excluded_reason is None and bodyonly.excluded_reason == "background"


# ---------- final review fixes ----------

def test_assign_heads_ranks_by_face_coverage_not_smallest_head():
    a = _cand((60, 40, 260, 700), face=(110, 110, 170, 180))    # facing person
    b = _cand((40, 20, 240, 700))                               # back-turned neighbour, no face
    a_head, b_head = D((90, 70, 200, 190), 0.9), D((100, 50, 175, 150), 0.9)   # B's smaller head also holds A's face centre
    pc.assign_heads([a, b], [a_head, b_head])
    assert a.head_box == (90, 70, 200, 190)
    assert b.head_box == (100, 50, 175, 150)


def test_assign_heads_oversized_head_does_not_beat_own_head():
    body = _cand((40, 20, 260, 700), face=(100, 100, 180, 200))   # facing person
    own_head = D((95, 60, 185, 190), 0.9)                          # covers 90% of the face
    stray_head = D((0, 0, 400, 300), 0.9)                          # covers 100% of the face, much larger
    pc.assign_heads([body], [own_head, stray_head])
    assert body.head_box == own_head.bbox


def test_exclusive_masks_length_mismatch_raises():
    m = _rect(10, 10, (0, 0, 5, 5))
    with pytest.raises(ValueError, match="same length"):
        pc.exclusive_masks([m, m], [None], [False, False])
    with pytest.raises(ValueError, match="same length"):
        pc.exclusive_masks([m], [None], [False, True])


def test_sam_prompts_skips_negative_inside_own_anchor():
    own_face = (100, 120, 200, 180)                             # centre (150, 150), covers x 140..200
    inside_own, elsewhere = (140, 120, 200, 180), (250, 120, 310, 180)   # centres (170, 150) and (280, 150)
    points, labels, _ = pc.sam_prompts((0, 0, 400, 800), own_face, [inside_own, elsewhere])
    assert points == [(150.0, 150.0), (280.0, 150.0)]
    assert labels == [1, 0]
    points, labels, _ = pc.sam_prompts((0, 0, 400, 800), None, [inside_own], target_head=own_face)
    assert points == [(150.0, 150.0)] and labels == [1]


# ---------- overlapping body ownership and SAM alternatives ----------

def test_overlapping_bodies_split_without_losing_unique_silhouette():
    a = _rect(100, 160, (0, 0, 100, 100))
    b = _rect(100, 160, (60, 0, 160, 100))
    originals = [a.copy(), b.copy()]
    out = pc.exclusive_masks([a, b], [(20, 5, 40, 25), (120, 5, 140, 25)], [False, False])
    assert not (out[0] & out[1]).any()
    assert out[0][:, :60].all() and out[1][:, 100:].all()
    assert out[0][50:, 60:80].all() and out[1][50:, 81:100].all()
    assert np.array_equal(out[0] | out[1], a | b)
    assert np.array_equal(a, originals[0]) and np.array_equal(b, originals[1])


def test_three_identical_group_masks_split_by_distinct_anchors():
    body = np.ones((60, 120), dtype=bool)
    faces = [(10, 0, 20, 10), (55, 0, 65, 10), (100, 0, 110, 10)]
    out = pc.exclusive_masks([body] * 3, faces, [False] * 3)
    assert np.array_equal(sum(m.astype(int) for m in out), np.ones(body.shape))
    assert all(m.any() for m in out)
    assert out[0][40, 15] and out[1][40, 60] and out[2][40, 105]
    # Numbering/sort order must not decide which person owns an overlap.
    reverse = pc.exclusive_masks([body] * 3, faces[::-1], [False] * 3)
    assert all(np.array_equal(a, b) for a, b in zip(out, reverse[::-1]))


def test_overlapping_anchors_still_have_single_pixel_owners():
    body = np.ones((60, 100), dtype=bool)
    out = pc.exclusive_masks([body, body], [(20, 5, 60, 25), (40, 5, 80, 25)], [False, False])
    assert not (out[0] & out[1]).any()
    assert out[0][15, 40] and out[1][15, 60]


def test_faceless_body_overlap_uses_mask_centres():
    a, b = _rect(50, 100, (0, 0, 70, 50)), _rect(50, 100, (30, 0, 100, 50))
    out = pc.exclusive_masks([a, b], [None, None], [False, False])
    assert not (out[0] & out[1]).any()
    assert out[0][25, 40] and out[1][25, 60]


def test_synthetic_rectangles_split_instead_of_erasing_each_other():
    body = np.ones((60, 100), dtype=bool)
    out = pc.exclusive_masks([body, body], [(10, 0, 20, 10), (80, 0, 90, 10)], [True, True])
    assert out[0][40, 15] and out[1][40, 85]
    assert not (out[0] & out[1]).any()


def test_fully_covered_synthetic_mask_stays_empty():
    real, synth = np.ones((50, 100), dtype=bool), _rect(50, 100, (30, 10, 70, 40))
    out = pc.exclusive_masks([real, synth], [None, None], [False, True])
    assert out[0].all() and not out[1].any()


def test_exclusive_masks_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="shape"):
        pc.exclusive_masks([np.ones((10, 10)), np.ones((20, 10))], [None, None], [False, False])


def test_sam_rejects_group_mask_and_keeps_single_body():
    own = _rect(60, 100, (5, 0, 45, 60))
    merged = own | _rect(60, 100, (55, 0, 95, 60))
    chosen = pc.select_sam_mask([merged, own], [(25, 10), (75, 10)], [1, 0], (0, 0, 100, 60), (60, 100))
    assert np.array_equal(chosen, own)


def test_sam_alternatives_are_not_unioned():
    a = _rect(60, 100, (5, 0, 45, 60))
    b = _rect(60, 100, (15, 0, 55, 50))
    chosen = pc.select_sam_mask([a, b], [(25, 10)], [1], (0, 0, 100, 60), (60, 100))
    assert np.array_equal(chosen, a)
    assert not chosen[:, 45:55].any()


def test_sam_prefers_full_body_over_face_only():
    face, body = _rect(60, 100, (10, 0, 30, 20)), _rect(60, 100, (0, 0, 40, 60))
    chosen = pc.select_sam_mask([face, body], [(20, 10)], [1], (0, 0, 100, 60), (60, 100))
    assert np.array_equal(chosen, body)


def test_sam_invalid_or_empty_candidates_fall_back():
    group = np.ones((60, 100), dtype=bool)
    for masks in ([], [group], [np.zeros_like(group)]):
        assert pc.select_sam_mask(masks, [(25, 10), (75, 10)], [1, 0], (0, 0, 100, 60), (60, 100)) is None


def test_sam_mask_is_clipped_to_target_body_box():
    full = np.ones((60, 100), dtype=bool)
    chosen = pc.select_sam_mask([full], [(20, 10)], [1], (10, 0, 40, 60), (60, 100))
    assert np.array_equal(chosen, _rect(60, 100, (10, 0, 40, 60)))


def test_sam_mask_shape_mismatch_raises():
    with pytest.raises(ValueError, match="SAM mask shape"):
        pc.select_sam_mask([np.ones((20, 30))], [(5, 5)], [1], (0, 0, 100, 60), (60, 100))


def test_face_association_uses_silhouettes_when_body_boxes_overlap():
    persons = [D((0, 0, 120, 200), .9), D((40, 0, 120, 160), .8)]
    faces = [D((60, 10, 80, 30), .9), D((90, 10, 110, 30), .8)]
    masks = [_rect(200, 120, (0, 0, 85, 200)), _rect(200, 120, (85, 0, 120, 160))]
    # Both faces fit the smaller box, but only one belongs to its silhouette.
    cands = pc.associate(persons, faces, 120, 200, person_masks=masks)
    assert len(cands) == 2
    assert [c.face_det for c in cands] == [0, 1]
    assert all(c.person_det is not None for c in cands)


def test_face_association_with_rectangles_keeps_bbox_behaviour():
    persons = [D((0, 0, 120, 200), .9), D((40, 0, 120, 160), .8)]
    faces = [D((60, 10, 80, 30), .9)]
    masks = [_rect(200, 120, p.bbox) for p in persons]
    cands = pc.associate(persons, faces, 120, 200, person_masks=masks)
    assert cands[1].face_det == 0 and cands[0].face_det is None


def test_face_association_rejects_mismatched_masks():
    persons = [D((0, 0, 10, 20), .9)]
    with pytest.raises(ValueError, match="person_masks"):
        pc.associate(persons, [], 10, 20, person_masks=[])
    with pytest.raises(ValueError, match="person_masks"):
        pc.associate(persons, [], 10, 20, person_masks=[np.ones((10, 10))])


def test_disjoint_silhouettes_keep_hair_inside_neighbours_face_margin():
    a = _rect(100, 120, (0, 0, 60, 100))
    b = _rect(100, 120, (60, 0, 120, 100))
    anchors = [(20, 20, 50, 50), (60, 20, 90, 50)]
    out = pc.exclusive_masks([a, b], anchors, [False, False], preserve_contours=True)
    assert np.array_equal(out[0], a) and np.array_equal(out[1], b)


def test_preserving_contours_still_removes_actual_face_and_body_overlap():
    a = _rect(100, 120, (0, 0, 80, 100))
    b = _rect(100, 120, (60, 0, 120, 100))
    out = pc.exclusive_masks([a, b], [(20, 10, 40, 30), (65, 10, 85, 30)],
                             [False, False], preserve_contours=True)
    assert not (out[0] & out[1]).any()
    assert not out[0][10:30, 65:80].any()
    assert out[1][10:30, 65:85].all()

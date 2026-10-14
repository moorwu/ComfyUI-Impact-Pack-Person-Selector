"""Reproduce SAM 3 person segmentation and verify Person Detector -> Selector.

Run in a SAM 3 environment with ComfyUI dependencies. For the verified hugging
photos, use --prompts man woman. CPU mode copies the SAM 3 package to the output
folder and guards three upstream CUDA-only setup operations; installed packages
are not edited. Images, masks, the comparison and a JSON report are saved locally.

Example:
  python sam3_person_check.py --image original.jpg --checkpoint sam3.pt \
    --sam3-root /path/to/sam3 --comfy-root /path/to/ComfyUI --output out --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def prepare_cpu_runtime(source: Path, output: Path) -> Path:
    package = source / 'sam3'
    destination = output / 'cpu_runtime' / 'sam3'
    shutil.copytree(package, destination, dirs_exist_ok=True)
    replacements = {
        'model/position_encoding.py': ('device="cuda"', 'device="cuda" if torch.cuda.is_available() else "cpu"'),
        'model/decoder.py': ('device="cuda"', 'device="cuda" if torch.cuda.is_available() else "cpu"'),
        'model/geometry_encoders.py': (
            'scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)',
            '(scale.pin_memory() if boxes_xyxy.is_cuda else scale).to(device=boxes_xyxy.device, non_blocking=True)',
        ),
    }
    for name, (old, new) in replacements.items():
        path = destination / name
        text = path.read_text()
        if new in text:
            continue
        if text.count(old) != 1:
            raise RuntimeError(f'CPU compatibility patch no longer matches {path}; inspect upstream changes')
        path.write_text(text.replace(old, new))
    return destination.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sam3-root', type=Path, required=True)
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--prompts', nargs='+', default=['man', 'woman'])
    parser.add_argument('--expected-people', type=int, default=2)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    if options.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        runtime = prepare_cpu_runtime(options.sam3_root, options.output)
    else:
        runtime = options.sam3_root
    sys.path[:0] = [str(runtime), str(options.comfy_root), str(Path(__file__).resolve().parents[2] / 'modules')]

    import numpy as np
    import torch
    from comfy.cli_args import args as comfy_args
    from PIL import Image, ImageDraw
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    from ultralytics import YOLO
    if options.device == 'cpu':
        comfy_args.cpu = True
    from impact import person_nodes as pn

    torch.set_num_threads(8)
    image_pil = Image.open(options.image).convert('RGB')
    pixels = np.array(image_pil)
    h, w = pixels.shape[:2]
    image = torch.from_numpy(pixels.astype(np.float32) / 255).unsqueeze(0)
    model = build_sam3_image_model(checkpoint_path=str(options.checkpoint), device=options.device, load_from_HF=False)
    processor = Sam3Processor(model, device=options.device, confidence_threshold=.5)
    detections = []
    with torch.inference_mode(), torch.autocast(options.device, dtype=torch.bfloat16):
        state = processor.set_image(image_pil)
        for prompt in options.prompts:
            processor.reset_all_prompts(state)
            result = processor.set_text_prompt(prompt, state)
            for probability, score in zip(result['masks_logits'], result['scores']):
                detections.append({'concept': prompt, 'confidence': float(score),
                                   'probability': probability.squeeze().float().cpu().numpy()})
    # Concept prompts may return the same instance; suppress by mask IoU, not body-box overlap.
    kept = []
    for detection in sorted(detections, key=lambda d: -d['confidence']):
        mask = detection['probability'] > .5
        if not mask.any():
            continue
        if any((mask & other['mask']).sum() / max((mask | other['mask']).sum(), 1) > .85 for other in kept):
            continue
        detection['mask'] = mask
        kept.append(detection)
    if len(kept) != options.expected_people:
        raise AssertionError(f'Expected {options.expected_people} people, got {len(kept)}')
    owner = np.stack([d['probability'] for d in kept]).argmax(axis=0)
    body_segs = []
    for i, detection in enumerate(kept):
        mask = detection['mask'] & (owner == i)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise AssertionError('An instance lost all its pixels')
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        detection.update(mask=mask, bbox=box)
        body_segs.append(pn._with_mask(pn._rect_seg(image, box, 3., 'person', detection['confidence']), mask))

    face_result = YOLO(str(options.comfy_root / 'models/ultralytics/bbox/face_yolov8m.pt'))(
        image_pil, conf=.5, device=options.device, verbose=False)[0]
    face_segs = [pn._rect_seg(image, tuple(int(v) for v in box), 3., 'face', float(score))
                 for box, score in zip(face_result.boxes.xyxy.tolist(), face_result.boxes.conf.tolist())]

    class CachedDetector:
        def __init__(self, items):
            self.items = items
        def detect(self, *args):
            return (h, w), self.items

    data, preview = pn.PersonDetectorSEGS().doit(
        image, CachedDetector(body_segs), CachedDetector(face_segs), threshold=.5,
        min_person_ratio=.015, min_relative_size=.31, min_face_size=24, min_sharpness=0.,
        sort_by='left_to_right', crop_factor=3., drop_size=10, person_segm_detector=CachedDetector(body_segs))
    people = [p for p in data['persons'] if p.index > 0]
    assert len(people) == options.expected_people
    report = {'image': str(options.image), 'prompts': options.prompts, 'people': [],
              'face_detections': len(face_segs), 'shared_pixels': int((sum(p.mask.astype(int) for p in people) > 1).sum())}
    assert report['shared_pixels'] == 0
    panels = [image_pil, Image.fromarray((preview[0].numpy() * 255).astype(np.uint8))]
    for person in people:
        original = next(d for d in kept if d['bbox'] == person.person_bbox)
        assert np.array_equal(person.mask, original['mask']), 'Node postprocessing damaged the instance silhouette'
        assert np.array_equal(pn._full_mask(person.person_seg, h, w), person.mask)
        selection = pn.PersonSelector().doit(data, image, str(person.index), 'any', '', False, .5, 0)['result']
        assert len(selection[1][1]) == 1
        assert np.array_equal(pn._full_mask(selection[1][1][0], h, w), person.mask)
        isolated = np.full(pixels.shape, 220, dtype=np.uint8)
        isolated[person.mask] = pixels[person.mask]
        panel = Image.fromarray(isolated)
        panel.save(options.output / f'person_{person.index}.png')
        Image.fromarray(person.mask.astype(np.uint8) * 255).save(options.output / f'mask_{person.index}.png')
        panels.append(panel)
        report['people'].append({'index': person.index, 'concept': original['concept'],
                                 'body_bbox': person.person_bbox, 'face_bbox': person.face_bbox,
                                 'pixels': int(person.mask.sum()), 'selector_seg_verified': True})
    canvas = Image.new('RGB', (384 * len(panels), 604), 'white')
    draw = ImageDraw.Draw(canvas)
    titles = ['Original', 'SAM3 -> Person Detector'] + [f'Person {p.index}' for p in people]
    for i, (panel, title) in enumerate(zip(panels, titles)):
        panel = panel.copy()
        panel.thumbnail((384, 576))
        canvas.paste(panel, (384 * i + (384 - panel.width) // 2, 28))
        draw.text((384 * i + 12, 8), title, fill='black')
    canvas.save(options.output / 'comparison.jpg', quality=95)
    (options.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

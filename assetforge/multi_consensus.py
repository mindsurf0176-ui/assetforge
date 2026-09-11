from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .frames import alpha_bbox, natural_key
from .path_safety import safe_output_child


def _paths(root: Path) -> list[Path]:
    paths = [p for p in root.rglob("*.png") if not p.name.startswith("_") and ".import" not in p.name]
    paths.sort(key=lambda p: natural_key(p.relative_to(root)))
    if len(paths) < 3:
        raise ValueError("multi-consensus needs at least three PNG frames")
    return paths


def _components(mask: np.ndarray, minimum: int = 8) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        queue = deque([(int(y), int(x))])
        points: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            points.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if len(points) >= minimum:
            component = np.zeros_like(mask, dtype=bool)
            ys, xs = zip(*points)
            component[ys, xs] = True
            result.append(component)
    return result


def multi_consensus(
    frames_dir: str | Path,
    output_dir: str | Path,
    *,
    clip: str | None = None,
    change_threshold: int = 24,
    min_component: int = 8,
) -> dict[str, Any]:
    """Align existing frames and measure stable motion regions without a model.

    This is an evidence stage for semantic decomposition. It never labels a
    component as an arm or weapon and therefore cannot falsely claim a rig is
    production-ready.
    """
    root = Path(frames_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"frame directory not found: {root}")
    paths = _paths(root)
    if clip:
        paths = [path for path in paths if path.stem == clip or path.stem.startswith(f"{clip}_")]
        if len(paths) < 3:
            raise ValueError(f"clip {clip!r} needs at least three PNG frames")
    loaded: list[tuple[np.ndarray, np.ndarray, tuple[int, int]]] = []
    for path in paths:
        with Image.open(path) as opened:
            image = np.asarray(opened.convert("RGBA"))
        box = alpha_bbox(Image.fromarray(image), 20)
        if box is None:
            continue
        left, top, right, bottom = box
        anchor = ((left + right - 1) // 2, bottom - 1)
        loaded.append((image[:, :, :3], image[:, :, 3] > 20, anchor))
    if len(loaded) < 3:
        raise ValueError("multi-consensus needs at least three non-empty frames")
    width = max(image.shape[1] for image, _mask, _anchor in loaded)
    height = max(image.shape[0] for image, _mask, _anchor in loaded)
    reference = loaded[len(loaded) // 2]
    ref_rgb, ref_mask, ref_anchor = reference
    motion = np.zeros((height, width), dtype=np.float32)
    aligned_count = 0
    for rgb, mask, anchor in loaded:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        alpha = np.zeros((height, width), dtype=bool)
        dx, dy = ref_anchor[0] - anchor[0], ref_anchor[1] - anchor[1]
        ys, xs = np.nonzero(mask)
        tx, ty = xs + dx, ys + dy
        valid = (tx >= 0) & (tx < width) & (ty >= 0) & (ty < height)
        canvas[ty[valid], tx[valid]] = rgb[ys[valid], xs[valid]]
        alpha[ty[valid], tx[valid]] = True
        ref = np.zeros_like(canvas)
        rh, rw = ref_rgb.shape[:2]
        ref[:rh, :rw] = ref_rgb
        reference_alpha = np.zeros((height, width), dtype=bool)
        reference_alpha[: ref_mask.shape[0], : ref_mask.shape[1]] = ref_mask
        comparable = alpha | reference_alpha
        delta = np.abs(canvas.astype(np.int16) - ref.astype(np.int16)).max(axis=2)
        motion += (comparable & (delta >= change_threshold)).astype(np.float32)
        aligned_count += 1
    confidence = motion / max(1, aligned_count)
    candidate_mask = confidence >= 0.25
    components = _components(candidate_mask, min_component)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    heat = np.zeros((height, width, 4), dtype=np.uint8)
    heat[:, :, 0] = np.clip(confidence * 255 * 2, 0, 255).astype(np.uint8)
    heat[:, :, 1] = np.clip((1 - confidence) * 90, 0, 90).astype(np.uint8)
    heat[:, :, 3] = np.where(confidence > 0, 220, 0).astype(np.uint8)
    heat_path = safe_output_child(output, "motion-heatmap.png", label="motion heatmap")
    Image.fromarray(heat).save(heat_path)
    masks_dir = safe_output_child(output, "candidate-masks", label="candidate masks")
    masks_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for index, component in enumerate(sorted(components, key=lambda item: int(item.sum()), reverse=True)):
        ys, xs = np.nonzero(component)
        mask = np.zeros((height, width, 4), dtype=np.uint8)
        mask[:, :, 3] = np.where(component, 255, 0).astype(np.uint8)
        path = safe_output_child(masks_dir, f"candidate_{index:02d}.png", label="candidate mask")
        Image.fromarray(mask).save(path)
        candidates.append({"file": str(path), "area": int(component.sum()), "bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]})
    report = {
        "schemaVersion": 1,
        "mode": "multi-frame-consensus",
        "sourceRoot": str(root),
        "clip": clip,
        "frameCount": len(loaded),
        "changeThreshold": change_threshold,
        "candidateThreshold": 0.25,
        "motionPixels": int(candidate_mask.sum()),
        "candidates": candidates,
        "automaticGate": {"production": False, "reason": "motion regions are evidence, not semantic body-part labels"},
        "heatmap": str(heat_path),
    }
    report_path = safe_output_child(output, "multi-consensus-report.json", label="consensus report")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return {"ok": True, "report": str(report_path), "heatmap": str(heat_path), "candidateCount": len(candidates), "frameCount": len(loaded)}

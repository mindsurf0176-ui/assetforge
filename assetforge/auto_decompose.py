from __future__ import annotations

import json
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image

from .frames import alpha_bbox
from .path_safety import safe_output_child
from .rig_build import autorig_reference


def _pngs(root: Path) -> list[Path]:
    paths = sorted(
        path for path in root.rglob("*.png")
        if not path.name.startswith("_") and path.name not in {"contact.png", "preview.png"}
    )
    if not paths:
        raise ValueError(f"no PNG frames found below {root}")
    return paths


def _choose_reference(paths: list[Path], root: Path) -> Path:
    preferred = root / "reference" / "east.png"
    if preferred.is_file():
        return preferred
    measurements: list[tuple[float, Path]] = []
    for path in paths:
        with Image.open(path) as opened:
            image = opened.convert("RGBA")
        box = alpha_bbox(image, 20)
        if box is None:
            continue
        left, top, right, bottom = box
        measurements.append((float((right - left) * (bottom - top)), path))
    if not measurements:
        raise ValueError("all input frames are empty")
    median_area = statistics.median(area for area, _path in measurements)
    return min(measurements, key=lambda item: abs(item[0] - median_area))[1]


def auto_decompose(
    frames_dir: str | Path,
    output_dir: str | Path,
    *,
    archetype: str,
    character: str,
    direction: str = "east",
    height: int = 192,
    clips: list[str] | None = None,
    resample: str = "nearest",
) -> dict[str, Any]:
    """Build a no-touch coarse rig and consistency report from existing frames.

    This is deliberately deterministic. It chooses a stable reference frame,
    delegates visible-pixel partitioning to the existing coarse auto-rig, and
    records all source-frame measurements for an automatic downstream gate.
    It never claims hidden-limb recovery or production semantic accuracy.
    """
    root = Path(frames_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"frame directory not found: {root}")
    paths = _pngs(root)
    reference = _choose_reference(paths, root)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = autorig_reference(
        reference,
        output,
        archetype=archetype,
        character=character,
        direction=direction,
        height=height,
        clips=clips,
        resample=resample,
    )
    rig_path = Path(report["rig"])
    rig = json.loads(rig_path.read_text(encoding="utf-8"))
    # The stock biped walk uses quarter-cycle keyframes, which produces
    # repeated rendered poses at an 8-frame sample. Expand it to a full
    # 8-contact cycle so automated output contains distinct passing/lift poses.
    walk = rig.get("clips", {}).get("walk")
    if walk is not None:
        times = [i / 8 for i in range(9)]
        def track(values: list[float]) -> list[list[float]]:
            return [[t, value] for t, value in zip(times, values)]
        walk["tracks"].update({
            "root": {"offset_y": track([0, -0.6, -1.5, -0.6, 0, -0.6, -1.5, -0.6, 0])},
            "spine": {"rotation": track([3.5, 4.5, 5.5, 4.8, 4.0, 4.8, 5.5, 4.5, 3.5])},
            "neck": {"rotation": track([-1, 0, 1, 0.5, -1, 0.5, 1, 0, -1])},
            "hip_f": {"rotation": track([24, 10, -10, -22, -24, -8, 10, 22, 24])},
            "hip_b": {"rotation": track([-24, -8, 10, 22, 24, 10, -8, -22, -24])},
            "sh_f": {"rotation": track([-20, -8, 8, 18, 18, 6, -10, -22, -20])},
            "sh_b": {"rotation": track([18, 8, -8, -18, -18, -6, 10, 22, 18])},
        })
        rig_path.write_text(json.dumps(rig, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    idle = rig.get("clips", {}).get("idle")
    if idle is not None:
        aim = deepcopy(idle)
        aim["fps"] = 8
        aim["loop"] = True
        aim["grounded"] = False
        aim["tracks"] = {
            "spine": {"rotation": [[0, 1.5], [0.35, 2.5], [0.7, 1.0], [1, 1.5]]},
            "neck": {"rotation": [[0, -1], [0.35, 0.5], [0.7, -0.5], [1, -1]]},
            "sh_f": {"rotation": [[0, 2], [0.35, 4.5], [0.7, 1], [1, 2]]},
            "sh_b": {"rotation": [[0, -1], [0.35, -3], [0.7, -0.5], [1, -1]]},
        }
        rig.setdefault("clips", {})["aim"] = aim
        rig_path.write_text(json.dumps(rig, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    measurements = []
    for path in paths:
        with Image.open(path) as opened:
            image = opened.convert("RGBA")
        box = alpha_bbox(image, 20)
        if box is None:
            measurements.append({"file": str(path), "empty": True})
            continue
        left, top, right, bottom = box
        measurements.append({
            "file": str(path),
            "size": list(image.size),
            "bbox": [left, top, right, bottom],
            "contentSize": [right - left, bottom - top],
            "opaquePixels": sum(1 for pixel in image.getchannel("A").getdata() if pixel > 20),
        })
    dataset = {
        "schemaVersion": 1,
        "mode": "auto-decompose",
        "quality": "coarse",
        "character": character,
        "direction": direction,
        "sourceRoot": str(root),
        "reference": str(reference),
        "frameCount": len(paths),
        "frames": measurements,
        "rig": str(rig_path),
        "walkPosePolicy": "distinct-contact-cycle",
        "automaticGate": {
            "technical": True,
            "production": False,
            "reason": "coarse semantic partition cannot verify hidden limbs or joint continuity",
        },
    }
    path = safe_output_child(output, "auto-decompose-report.json", label="auto-decompose report")
    path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "quality": "coarse", "report": str(path), "rig": str(rig_path), "reference": str(reference), "frameCount": len(paths)}

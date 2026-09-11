from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .json_utils import strict_json_loads
from .path_safety import safe_output_child
from .rig_build import _slot_contract


def _mapping(value: str | Path) -> dict[str, str]:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"vision mapping not found: {path}")
    data = strict_json_loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("parts"), dict):
        data = data["parts"]
    if not isinstance(data, dict) or not data:
        raise ValueError("vision mapping must be a non-empty object of slot to PNG")
    result: dict[str, str] = {}
    for slot, raw in data.items():
        if not isinstance(slot, str) or not isinstance(raw, str):
            raise ValueError("vision mapping entries must be string slot/file pairs")
        result[slot] = raw
    return result


def stage_vision_layers(
    layers_dir: str | Path,
    mapping_file: str | Path,
    output_dir: str | Path,
    *,
    archetype: str,
) -> dict[str, Any]:
    """Materialize a model-produced layer manifest as AssetForge part PNGs.

    The external model owns segmentation. AssetForge owns slot validation and
    rendering, so an incomplete or ambiguous manifest fails before animation.
    """
    layers = Path(layers_dir).expanduser().resolve()
    if not layers.is_dir():
        raise FileNotFoundError(f"vision layer directory not found: {layers}")
    contract = _slot_contract(archetype)
    mapping = _mapping(mapping_file)
    unknown = sorted(set(mapping) - set(contract))
    if unknown:
        raise ValueError(f"vision mapping has unsupported slots: {', '.join(unknown)}")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for slot, raw_file in mapping.items():
        source = (layers / raw_file).resolve()
        if layers not in source.parents:
            raise ValueError(f"vision layer escapes input directory: {raw_file!r}")
        if not source.is_file() or source.suffix.lower() != ".png":
            raise FileNotFoundError(f"vision layer PNG not found: {source}")
        target = safe_output_child(output, f"{slot}.png", label="vision part")
        with Image.open(source) as opened:
            image = opened.convert("RGBA")
            if image.getchannel("A").getbbox() is None:
                raise ValueError(f"vision layer is empty: {source.name}")
            image.save(target)
        staged.append(slot)
    report = {
        "schemaVersion": 1,
        "mode": "vision-import",
        "source": "external-layer-manifest",
        "archetype": archetype,
        "layers": sorted(staged),
        "partsDir": str(output),
        "mapping": str(Path(mapping_file).expanduser().resolve()),
    }
    report_path = safe_output_child(output.parent, "vision-import-report.json", label="vision import report")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return report

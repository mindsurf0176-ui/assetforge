"""Identity-locked clip generation graph for AssetForge.

Generators still produce pixels. This module owns the job DAG, layout guides,
pre-ingest QA, and the smallest repair scope. It does not call an image model.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .frames import make_contact_sheet, split_source_sheet
from .json_utils import strict_json_loads
from .local_animation import parse_clips, parse_frame_counts
from .path_safety import reset_output_directory, safe_output_child
from .profile import Profile
from .rig_core import RigError


SCHEMA_VERSION = 1
KIND = "assetforge-clip-run"
JOBS_NAME = "clip-jobs.json"
CANONICAL_BASE = "references/canonical-base.png"
LAYOUT_GUIDE_DIR = "references/layout-guides"
PROMPT_DIR = "prompts"
DECODED_DIR = "decoded"
FRAMES_DIR = "frames"
QA_DIR = "qa"

PREFERRED_CLIPS = ("idle", "walk", "aim", "attack", "hit", "death")
CLIP_PURPOSES = {
    "idle": "combat-ready idle with subtle breathing; no locomotion or new props",
    "walk": "contact, passing, and opposite-contact gait with feet on the shared ground line",
    "aim": "hold aim with minimal breathing sway; equipment stays attached",
    "attack": "anticipation, action, recoil, and recovery",
    "hit": "brief hit flinch; equipment stays attached",
    "death": "stagger, fall, and a final grounded pose",
}

GUIDE_BACKGROUND = (138, 0, 255)
GUIDE_MARK = (255, 0, 153)
GUIDE_COLOR_DISTANCE = 16.0
GUIDE_BACKGROUND_RATIO_LIMIT = 0.01
GUIDE_MARK_PIXEL_LIMIT = 40
IDENTITY_MIN_PALETTE_OVERLAP = 0.42
IDENTITY_MIN_COLOR_SIMILARITY = 0.72
MIN_FOREGROUND_PIXELS = 16


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _run_root(run_dir: str | Path) -> Path:
    root = Path(run_dir).expanduser()
    if root.is_symlink():
        raise ValueError(f"clip-run directory must not be a symbolic link: {root}")
    return root.resolve()


def _jobs_path(root: Path) -> Path:
    return safe_output_child(root, JOBS_NAME, label="clip-run manifest")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if value.endswith("\n") else value + "\n", encoding="utf-8")


def _split_rel(*parts: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for part in parts:
        tokens.extend(segment for segment in str(part).split("/") if segment and segment != ".")
    return tuple(tokens)


def load_clip_run(run_dir: str | Path) -> dict[str, Any]:
    root = _run_root(run_dir)
    path = root / JOBS_NAME
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"clip-run manifest not found: {path}")
    data = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("kind") != KIND:
        raise ValueError(f"not an AssetForge clip-run manifest: {path}")
    data["_root"] = str(root)
    return data


def save_clip_run(manifest: dict[str, Any]) -> Path:
    root = _run_root(manifest["_root"])
    payload = {key: value for key, value in manifest.items() if key != "_root"}
    path = _jobs_path(root)
    _write_json(path, payload)
    return path


def job_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("clip-run manifest jobs must be a list")
    result: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, dict) or not job.get("id"):
            raise ValueError("each clip-run job must be an object with an id")
        result[str(job["id"])] = job
    return result


def ready_jobs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = job_map(manifest)
    ready: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        if job.get("status") == "complete":
            continue
        depends = job.get("depends_on") or []
        if any(jobs[str(job_id)].get("status") != "complete" for job_id in depends):
            continue
        ready.append(job)
    return ready


def _sheet_grid(frame_count: int) -> tuple[int, int]:
    if frame_count <= 8:
        return frame_count, 1
    if frame_count % 4 == 0:
        return 4, frame_count // 4
    if frame_count % 3 == 0:
        return 3, frame_count // 3
    columns = min(6, frame_count)
    return columns, math.ceil(frame_count / columns)


def _ordered_clips(profile: Profile, requested: list[str] | None) -> list[str]:
    available = list(profile.data.get("animations", {}))
    if requested is None:
        chosen = [name for name in PREFERRED_CLIPS if name in available]
        chosen.extend(name for name in available if name not in chosen)
    else:
        unknown = [name for name in requested if name not in available]
        if unknown:
            raise ValueError(
                f"profile {profile.id!r} has no animation {unknown[0]!r}; "
                f"choose one of {sorted(available)}"
            )
        chosen = list(dict.fromkeys(requested))
    if not chosen:
        raise ValueError(f"profile {profile.id!r} has no animations to schedule")
    return chosen


def _resolved_frame_count(
    profile: Profile,
    clip: str,
    overrides: dict[str, int],
) -> int:
    contract = profile.animation(clip)
    minimum = int(contract["minFrames"])
    maximum = int(contract["maxFrames"])
    count = int(overrides.get(clip, maximum))
    if count < minimum or count > maximum:
        raise ValueError(
            f"{profile.id}:{clip} frame count {count} is outside {minimum}..{maximum}"
        )
    return count


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    dash: int = 6,
    gap: int = 4,
) -> None:
    x1, y1 = start
    x2, y2 = end
    if x1 == x2:
        for y in range(min(y1, y2), max(y1, y2) + 1, dash + gap):
            draw.line((x1, y, x2, min(y + dash - 1, max(y1, y2))), fill=fill)
        return
    if y1 == y2:
        for x in range(min(x1, x2), max(x1, x2) + 1, dash + gap):
            draw.line((x, y1, min(x + dash - 1, max(x1, x2)), y2), fill=fill)
        return
    raise ValueError("layout guides only use horizontal or vertical construction lines")


def create_layout_guide(
    path: Path,
    *,
    clip: str,
    frames: int,
    columns: int,
    rows: int,
    cell: tuple[int, int],
    ground_y: int,
) -> dict[str, Any]:
    cell_w, cell_h = cell
    image = Image.new("RGB", (columns * cell_w, rows * cell_h), GUIDE_BACKGROUND)
    draw = ImageDraw.Draw(image)
    for index in range(frames):
        column = index % columns
        row = index // columns
        left = column * cell_w
        top = row * cell_h
        right = left + cell_w - 1
        bottom = top + cell_h - 1
        margin = max(3, min(cell_w, cell_h) // 12)
        draw.rectangle((left, top, right, bottom), outline=GUIDE_MARK, width=2)
        draw.rectangle(
            (left + margin, top + margin, right - margin, bottom - margin),
            outline=GUIDE_MARK,
            width=1,
        )
        center_x = left + cell_w // 2
        line_y = top + min(max(ground_y, margin), cell_h - margin - 1)
        _draw_dashed_line(draw, (left + margin, line_y), (right - margin, line_y), fill=GUIDE_MARK)
        _draw_dashed_line(
            draw,
            (center_x, top + margin),
            (center_x, bottom - margin),
            fill=GUIDE_MARK,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return {
        "clip": clip,
        "path": path.name,
        "frames": frames,
        "columns": columns,
        "rows": rows,
        "cell": [cell_w, cell_h],
        "groundY": ground_y,
        "usage": "layout construction only; never copy guide pixels into generated frames",
    }


def _copy_rgba(source: str | Path, destination: Path) -> Path:
    src = Path(source).expanduser()
    if src.is_symlink() or not src.is_file():
        raise FileNotFoundError(f"image not found: {src}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as opened:
        opened.convert("RGBA").save(destination)
    return destination


def _color_distance(pixels: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    return np.linalg.norm(pixels.astype(np.float32) - np.asarray(color, dtype=np.float32), axis=2)


def _opaque_rgb(image: Image.Image, min_alpha: int = 20) -> np.ndarray:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return rgba[rgba[:, :, 3] > min_alpha][:, :3]


def _quantized_palette(rgb: np.ndarray, step: int = 16) -> set[tuple[int, int, int]]:
    if rgb.size == 0:
        return set()
    quantized = (rgb.astype(np.int16) // step) * step
    return {tuple(int(channel) for channel in row) for row in np.unique(quantized, axis=0)}


def _palette_overlap(left: set[tuple[int, int, int]], right: set[tuple[int, int, int]]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union)


def _mean_color(rgb: np.ndarray) -> np.ndarray:
    if rgb.size == 0:
        return np.zeros(3, dtype=np.float32)
    return rgb.astype(np.float32).mean(axis=0)


def inspect_guide_pixels(image: Image.Image) -> dict[str, Any]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    total = int(rgb.shape[0] * rgb.shape[1])
    background = int(np.count_nonzero(_color_distance(rgb, GUIDE_BACKGROUND) <= GUIDE_COLOR_DISTANCE))
    marks = int(np.count_nonzero(_color_distance(rgb, GUIDE_MARK) <= GUIDE_COLOR_DISTANCE))
    ratio = background / total if total else 0.0
    errors: list[str] = []
    if ratio > GUIDE_BACKGROUND_RATIO_LIMIT:
        errors.append(
            f"layout-guide background covers {ratio:.3f} of the image; "
            "construction pixels must not appear in generated output"
        )
    if marks > GUIDE_MARK_PIXEL_LIMIT:
        errors.append(
            f"layout-guide marks cover {marks} pixels; construction lines must not be copied"
        )
    return {
        "backgroundPixels": background,
        "backgroundRatio": round(ratio, 4),
        "markPixels": marks,
        "errors": errors,
    }


def inspect_identity(frame: Image.Image, canonical: Image.Image, *, clip: str) -> dict[str, Any]:
    frame_rgb = _opaque_rgb(frame)
    canonical_rgb = _opaque_rgb(canonical)
    frame_palette = _quantized_palette(frame_rgb)
    canonical_palette = _quantized_palette(canonical_rgb)
    overlap = _palette_overlap(frame_palette, canonical_palette)
    color_distance = float(np.linalg.norm(_mean_color(frame_rgb) - _mean_color(canonical_rgb)))
    color_similarity = max(0.0, 1.0 - color_distance / math.sqrt(3.0 * 255.0**2))
    errors: list[str] = []
    if len(frame_rgb) < MIN_FOREGROUND_PIXELS:
        errors.append("frame has no usable foreground against the canonical identity")
    if overlap < IDENTITY_MIN_PALETTE_OVERLAP:
        errors.append(
            f"{clip} palette overlap {overlap:.3f} is below {IDENTITY_MIN_PALETTE_OVERLAP:.2f}; "
            "identity drift is a blocker even when geometry validation is clean"
        )
    if color_similarity < IDENTITY_MIN_COLOR_SIMILARITY:
        errors.append(
            f"{clip} mean-color similarity {color_similarity:.3f} is below "
            f"{IDENTITY_MIN_COLOR_SIMILARITY:.2f}"
        )
    return {
        "paletteOverlap": round(overlap, 4),
        "colorSimilarity": round(color_similarity, 4),
        "foregroundPixels": int(len(frame_rgb)),
        "errors": errors,
    }


def _clip_prompt(
    *,
    character: str,
    clip: str,
    direction: str,
    frames: int,
    columns: int,
    rows: int,
    cell: tuple[int, int],
    purpose: str,
    style: str,
    animation_prompt: str,
) -> str:
    cell_w, cell_h = cell
    extra = f" {animation_prompt.strip()}" if animation_prompt.strip() else ""
    return f"""Create one complete {clip} animation sheet for {character}, facing {direction}.

Use the attached canonical base as the locked identity. Use the attached layout guide only for slot count, spacing, centering, padding, and the ground line. Do not draw the guide, its colors, boxes, center marks, or labels.

Output exactly {frames} unique full-body poses in a grid of {columns} columns and {rows} rows of {cell_w}x{cell_h} cells, read left to right then top to bottom. One complete unclipped pose per cell. Feet rest on the shared ground line. Keep camera, scale, silhouette, face, hair, costume, weapons, attachments, and palette identical to the canonical base.

Motion: {purpose}.{extra}
Style: {style}
Background: genuine transparent RGBA, or one flat saturated chroma key that is not a construction color.
Constraints: dense authored pixel clusters, nearest-neighbor edges, no anti-aliasing, no scenery, no shadows as separate sprites, no text, no UI.
Forbidden: copying layout-guide pixels, duplicated holds, identity drift, missing equipment, cropped bodies, slot overlap.

After generation, return only:
selected_source=/absolute/path/to/selected-output.png
qa_note=<one sentence>
Do not ingest, edit clip-jobs.json, or package the result.
"""


def _base_prompt(*, character: str, direction: str, style: str, locked: bool) -> str:
    action = (
        "Normalize and preserve this locked identity as a single full-body canonical still."
        if locked
        else "Create the canonical full-body identity still for later clip generation."
    )
    return f"""{action}

Character: {character}, facing {direction}.
Style: {style}
Output one centered full-body character on genuine transparent RGBA or a flat chroma key. No scenery, text, UI, shadows, or layout guides.
This image becomes the identity lock for every later clip. Do not invent a second costume, weapon, or silhouette.

After generation, return only:
selected_source=/absolute/path/to/selected-output.png
qa_note=<one sentence>
Do not ingest, edit clip-jobs.json, or package the result.
"""


def _clip_depends_on(clip: str, ordered: list[str]) -> list[str]:
    if clip == "idle" or "idle" not in ordered:
        return ["base"]
    return ["base", "idle"]


def prepare_clip_run(
    profile: Profile,
    *,
    character: str,
    tier: str,
    direction: str,
    output: str | Path,
    clips: list[str] | str | None = None,
    frames: str | dict[str, int] | None = None,
    reference: str | Path | None = None,
    lock_reference: bool = False,
    cell: tuple[int, int] | None = None,
    provider: str = "external",
    force: bool = False,
) -> dict[str, Any]:
    """Create a clip-run folder, layout guides, prompts, and a job DAG."""

    try:
        requested = None if clips is None else parse_clips(clips)
        overrides = frames if isinstance(frames, dict) else parse_frame_counts(frames)
    except RigError as exc:
        raise ValueError(str(exc)) from exc
    ordered = _ordered_clips(profile, requested)
    unknown_overrides = sorted(set(overrides) - set(ordered))
    if unknown_overrides:
        raise ValueError(
            f"frame overrides target unscheduled clips: {', '.join(unknown_overrides)}"
        )
    tier_data = profile.tier(tier)
    if tier_data.get("canvasPolicy") != "fixed" or not tier_data.get("canvas"):
        raise ValueError("clip-run requires a fixed-canvas profile tier")
    canvas = (int(tier_data["canvas"][0]), int(tier_data["canvas"][1]))
    cell_size = cell or canvas
    anchor = tier_data.get("anchor") or [cell_size[0] // 2, cell_size[1] - 1]
    ground_y = int(round(int(anchor[1]) * cell_size[1] / canvas[1]))
    generation = profile.data.get("generation", {})
    style = ", ".join(
        part
        for part in (
            generation.get("projection", ""),
            generation.get("stylePrompt", "authored native pixel art"),
        )
        if part
    )

    output_path = Path(output).expanduser()
    if output_path.exists() and (output_path / JOBS_NAME).is_file() and not force:
        raise ValueError(
            f"clip-run already exists: {output_path.resolve()}; pass --force to replace it"
        )
    root = reset_output_directory(output, label="clip-run output")

    guide_dir = safe_output_child(root, *_split_rel(LAYOUT_GUIDE_DIR), label="layout guide directory")
    prompt_dir = safe_output_child(root, PROMPT_DIR, label="prompt directory")
    decoded_dir = safe_output_child(root, DECODED_DIR, label="decoded directory")
    frames_root = safe_output_child(root, FRAMES_DIR, label="frames directory")
    qa_dir = safe_output_child(root, QA_DIR, label="qa directory")
    for path in (guide_dir, prompt_dir, decoded_dir, frames_root, qa_dir):
        path.mkdir(parents=True, exist_ok=True)

    reference_path = Path(reference).expanduser().resolve() if reference else None
    if reference and (reference_path is None or reference_path.is_symlink() or not reference_path.is_file()):
        raise FileNotFoundError(f"clip-run reference not found: {reference}")
    if lock_reference and reference_path is None:
        raise ValueError("--lock-reference requires --reference")

    copied_reference = None
    if reference_path is not None:
        copied_reference = _copy_rgba(
            reference_path,
            safe_output_child(root, "references", "source-reference.png", label="source reference"),
        )

    base_inputs = []
    if copied_reference is not None:
        base_inputs.append({"path": _rel(copied_reference, root), "role": "identity reference"})
    base_prompt_path = safe_output_child(root, PROMPT_DIR, "base.md", label="base prompt")
    _write_text(
        base_prompt_path,
        _base_prompt(
            character=character,
            direction=direction,
            style=style,
            locked=lock_reference,
        ),
    )
    jobs: list[dict[str, Any]] = [
        {
            "id": "base",
            "kind": "canonical-base",
            "status": "pending",
            "clip": None,
            "depends_on": [],
            "frame_count": 1,
            "columns": 1,
            "rows": 1,
            "cell": list(cell_size),
            "prompt_file": _rel(base_prompt_path, root),
            "input_images": base_inputs,
            "output_path": f"{DECODED_DIR}/base.png",
            "canonical_path": CANONICAL_BASE,
            "requires_grounded_generation": bool(base_inputs) and not lock_reference,
            "allow_prompt_only_generation": not base_inputs,
            "derivation_policy": {"mirror": False},
            "parallelizable_after": [],
        }
    ]

    for clip in ordered:
        count = _resolved_frame_count(profile, clip, overrides)
        columns, rows = _sheet_grid(count)
        guide = create_layout_guide(
            safe_output_child(root, *_split_rel(LAYOUT_GUIDE_DIR, f"{clip}.png"), label="layout guide"),
            clip=clip,
            frames=count,
            columns=columns,
            rows=rows,
            cell=cell_size,
            ground_y=ground_y,
        )
        prompt_path = safe_output_child(root, PROMPT_DIR, f"{clip}.md", label="clip prompt")
        contract = profile.animation(clip)
        _write_text(
            prompt_path,
            _clip_prompt(
                character=character,
                clip=clip,
                direction=direction,
                frames=count,
                columns=columns,
                rows=rows,
                cell=cell_size,
                purpose=CLIP_PURPOSES.get(clip, clip),
                style=style,
                animation_prompt=str(contract.get("prompt", "")),
            ),
        )
        depends_on = _clip_depends_on(clip, ordered)
        jobs.append(
            {
                "id": clip,
                "kind": "clip-sheet",
                "status": "pending",
                "clip": clip,
                "depends_on": depends_on,
                "frame_count": count,
                "columns": columns,
                "rows": rows,
                "cell": list(cell_size),
                "prompt_file": _rel(prompt_path, root),
                "layout_guide": f"{LAYOUT_GUIDE_DIR}/{guide['path']}",
                "input_images": [
                    {"path": CANONICAL_BASE, "role": "canonical identity reference"},
                    {
                        "path": f"{LAYOUT_GUIDE_DIR}/{guide['path']}",
                        "role": "layout construction guide; do not copy",
                    },
                ],
                "output_path": f"{DECODED_DIR}/{clip}.png",
                "frames_dir": f"{FRAMES_DIR}/{clip}",
                "requires_grounded_generation": True,
                "allow_prompt_only_generation": False,
                "derivation_policy": {"mirror": False},
                "parallelizable_after": list(depends_on),
                "contract": {
                    "minFrames": int(contract["minFrames"]),
                    "maxFrames": int(contract["maxFrames"]),
                    "fps": contract["fps"],
                    "loop": contract["loop"],
                    "canvas": list(canvas),
                    "anchor": [int(anchor[0]), int(anchor[1])],
                },
            }
        )

    manifest: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": KIND,
        "profile": profile.id,
        "profileFingerprint": profile.fingerprint,
        "profilePath": str(profile.path),
        "character": character,
        "tier": tier,
        "direction": direction,
        "provider": provider,
        "cell": list(cell_size),
        "anchor": [int(anchor[0]), int(anchor[1])],
        "groundY": ground_y,
        "canonicalBase": CANONICAL_BASE,
        "lockReference": bool(lock_reference),
        "constructionColors": {
            "background": list(GUIDE_BACKGROUND),
            "mark": list(GUIDE_MARK),
            "maxBackgroundRatio": GUIDE_BACKGROUND_RATIO_LIMIT,
            "maxMarkPixels": GUIDE_MARK_PIXEL_LIMIT,
        },
        "identity": {
            "minPaletteOverlap": IDENTITY_MIN_PALETTE_OVERLAP,
            "minColorSimilarity": IDENTITY_MIN_COLOR_SIMILARITY,
            "driftIsBlocker": True,
        },
        "workerProtocol": {
            "return": ["selected_source", "qa_note"],
            "parentInspects": [f"{QA_DIR}/contact-sheet.png", f"{QA_DIR}/previews"],
            "doNotAttachEveryGeneratedPngToParent": True,
        },
        "jobs": jobs,
        "_root": str(root),
    }

    accepted = None
    if lock_reference:
        accepted = accept_job(manifest, "base", source=copied_reference, qa_note="locked supplied reference")

    save_clip_run(manifest)
    result = {
        "ok": True if accepted is None else bool(accepted["ok"]),
        "run": str(root),
        "manifest": str(_jobs_path(root)),
        "canonicalBase": str(safe_output_child(root, *_split_rel(CANONICAL_BASE), label="canonical base")),
        "readyJobs": [job["id"] for job in ready_jobs(manifest)],
        "jobs": [
            {
                "id": job["id"],
                "status": job["status"],
                "depends_on": job["depends_on"],
            }
            for job in manifest["jobs"]
        ],
        "workerProtocol": manifest["workerProtocol"],
    }
    result["canonicalBaseExists"] = Path(result["canonicalBase"]).is_file()
    if accepted is not None:
        result["base"] = accepted
    return result


def _job_output(root: Path, job: dict[str, Any]) -> Path:
    return safe_output_child(root, *_split_rel(str(job["output_path"])), label="job output")


def _frame_paths(root: Path, job: dict[str, Any]) -> list[Path]:
    frames_dir = safe_output_child(root, *_split_rel(str(job["frames_dir"])), label="clip frames")
    return sorted(path for path in frames_dir.glob("*.png") if path.is_file() and not path.is_symlink())


def _inspect_clip_frames(
    frames: list[Path],
    canonical: Image.Image,
    clip: str,
) -> dict[str, Any]:
    reports = []
    errors: list[str] = []
    failed_frames: list[int] = []
    for index, path in enumerate(frames):
        with Image.open(path) as opened:
            frame = opened.convert("RGBA")
        guide = inspect_guide_pixels(frame)
        identity = inspect_identity(frame, canonical, clip=clip)
        frame_errors = list(guide["errors"]) + list(identity["errors"])
        if frame_errors:
            failed_frames.append(index)
            errors.extend(f"{clip} frame {index}: {message}" for message in frame_errors)
        reports.append(
            {
                "index": index,
                "path": path.name,
                "guide": guide,
                "identity": {key: value for key, value in identity.items() if key != "errors"},
                "errors": frame_errors,
            }
        )
    return {
        "frames": reports,
        "failedFrames": failed_frames,
        "errors": errors,
    }


def _write_preview_gif(frames: list[Path], output: Path, *, fps: float) -> Path:
    images = [Image.open(path).convert("RGBA") for path in frames]
    duration = max(40, int(round(1000 / max(fps, 0.001))))
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )
    for image in images:
        image.close()
    return output


def inspect_job(manifest: dict[str, Any], job_id: str) -> dict[str, Any]:
    root = _run_root(manifest["_root"])
    job = job_map(manifest)[job_id]
    result: dict[str, Any] = {
        "id": job_id,
        "kind": job["kind"],
        "errors": [],
    }
    if job["kind"] == "canonical-base":
        output = _job_output(root, job)
        if not output.is_file():
            raise FileNotFoundError(f"job {job_id!r} has no decoded output: {output}")
        with Image.open(output) as opened:
            decoded = opened.convert("RGBA")
        errors = list(inspect_guide_pixels(decoded)["errors"])
        if len(_opaque_rgb(decoded)) < MIN_FOREGROUND_PIXELS:
            errors.append("canonical base has no usable foreground")
        result["guide"] = inspect_guide_pixels(decoded)
        result["errors"] = list(dict.fromkeys(errors))
        return result

    canonical_path = safe_output_child(root, *_split_rel(CANONICAL_BASE), label="canonical base")
    if not canonical_path.is_file():
        result["errors"] = ["canonical base is locked before clip inspection"]
        return result
    frame_paths = _frame_paths(root, job)
    errors: list[str] = []
    if len(frame_paths) != int(job["frame_count"]):
        errors.append(
            f"{job_id} has {len(frame_paths)} extracted frames; expected {job['frame_count']}"
        )
    with Image.open(canonical_path) as opened:
        canonical = opened.convert("RGBA")
    clip_report = _inspect_clip_frames(frame_paths, canonical, str(job["clip"]))
    errors.extend(clip_report["errors"])
    result["frames"] = clip_report["frames"]
    result["failedFrames"] = clip_report["failedFrames"]
    result["errors"] = list(dict.fromkeys(errors))
    return result


def _refresh_qa(manifest: dict[str, Any]) -> dict[str, Any]:
    root = _run_root(manifest["_root"])
    preview_dir = safe_output_child(root, QA_DIR, "previews", label="preview directory")
    preview_dir.mkdir(parents=True, exist_ok=True)
    contact_paths: list[Path] = []
    previews: list[str] = []
    for job in manifest["jobs"]:
        if job.get("kind") != "clip-sheet" or job.get("status") != "complete":
            continue
        frame_paths = _frame_paths(root, job)
        if not frame_paths:
            continue
        contact_paths.extend(frame_paths)
        fps = float((job.get("contract") or {}).get("fps") or 8)
        preview = _write_preview_gif(
            frame_paths,
            safe_output_child(root, QA_DIR, "previews", f"{job['id']}.gif", label="clip preview"),
            fps=fps,
        )
        previews.append(_rel(preview, root))
    contact = None
    if contact_paths:
        contact_path = make_contact_sheet(
            contact_paths,
            safe_output_child(root, QA_DIR, "contact-sheet.png", label="contact sheet"),
            scale=2,
        )
        contact = _rel(contact_path, root)
    qa = {
        "contactSheet": contact,
        "previews": previews,
        "parentInspectsOnly": True,
    }
    _write_json(safe_output_child(root, QA_DIR, "review.json", label="qa review"), qa)
    return qa


def accept_job(
    manifest: dict[str, Any],
    job_id: str,
    *,
    source: str | Path | None = None,
    source_dir: str | Path | None = None,
    frame: int | None = None,
    qa_note: str | None = None,
) -> dict[str, Any]:
    """Copy a generated result into the run and inspect it before marking complete."""

    root = _run_root(manifest["_root"])
    jobs = job_map(manifest)
    if job_id not in jobs:
        raise ValueError(f"unknown clip-run job: {job_id}")
    job = jobs[job_id]
    missing = [
        str(dep)
        for dep in job.get("depends_on") or []
        if jobs[str(dep)].get("status") != "complete"
    ]
    if missing:
        raise ValueError(f"job {job_id!r} is blocked by incomplete dependencies: {', '.join(missing)}")
    if source is None and source_dir is None:
        raise ValueError("accept requires --source or --source-dir")
    if source is not None and source_dir is not None:
        raise ValueError("accept takes either --source or --source-dir, not both")

    decoded = _job_output(root, job)
    if job["kind"] == "canonical-base":
        if source is None:
            raise ValueError("canonical base accept requires --source")
        _copy_rgba(source, decoded)
        canonical = _copy_rgba(
            decoded,
            safe_output_child(root, *_split_rel(CANONICAL_BASE), label="canonical base"),
        )
        report = inspect_job(manifest, job_id)
        job["source_path"] = str(Path(source).expanduser().resolve())
        job["completed_at"] = _utc_now()
        job["inspection"] = report
        if qa_note:
            job["qa_note"] = qa_note
        job["status"] = "complete" if not report["errors"] else "failed"
        save_clip_run(manifest)
        return {
            "ok": job["status"] == "complete",
            "job": job_id,
            "status": job["status"],
            "canonicalBase": str(canonical),
            "errors": report["errors"],
            "readyJobs": [item["id"] for item in ready_jobs(manifest)],
        }

    frames_dir = safe_output_child(root, *_split_rel(str(job["frames_dir"])), label="clip frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    if frame is not None:
        if source is None:
            raise ValueError("frame repair requires --source")
        if frame < 0 or frame >= int(job["frame_count"]):
            raise ValueError(f"frame {frame} is outside 0..{int(job['frame_count']) - 1}")
        target = safe_output_child(frames_dir, f"frame_{frame:02d}.png", label="repaired frame")
        _copy_rgba(source, target)
    elif source_dir is not None:
        directory = Path(source_dir).expanduser().resolve()
        if directory.is_symlink() or not directory.is_dir():
            raise FileNotFoundError(f"frame directory not found: {directory}")
        incoming = sorted(
            path
            for path in directory.iterdir()
            if path.suffix.lower() == ".png" and path.is_file() and not path.is_symlink()
        )
        if len(incoming) != int(job["frame_count"]):
            raise ValueError(
                f"{job_id} source directory has {len(incoming)} PNG files; expected {job['frame_count']}"
            )
        for index, path in enumerate(incoming):
            _copy_rgba(path, safe_output_child(frames_dir, f"frame_{index:02d}.png", label="clip frame"))
        first = Image.open(incoming[0]).convert("RGBA")
        sheet = Image.new(
            "RGBA",
            (int(job["columns"]) * first.width, int(job["rows"]) * first.height),
            (0, 0, 0, 0),
        )
        for index, path in enumerate(incoming):
            with Image.open(path) as opened:
                frame_image = opened.convert("RGBA")
            column = index % int(job["columns"])
            row = index // int(job["columns"])
            sheet.alpha_composite(frame_image, (column * first.width, row * first.height))
        decoded.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(decoded)
        first.close()
    else:
        _copy_rgba(source, decoded)
        split_source_sheet(
            decoded,
            frames_dir,
            columns=int(job["columns"]),
            rows=int(job["rows"]),
            frame_count=int(job["frame_count"]),
            prefix="frame",
        )

    report = inspect_job(manifest, job_id)
    job["source_path"] = str(Path(source or source_dir).expanduser().resolve())
    job["completed_at"] = _utc_now()
    job["inspection"] = report
    if qa_note:
        job["qa_note"] = qa_note
    job["status"] = "complete" if not report["errors"] else "failed"
    qa = _refresh_qa(manifest)
    save_clip_run(manifest)
    return {
        "ok": job["status"] == "complete",
        "job": job_id,
        "status": job["status"],
        "failedFrames": report.get("failedFrames", []),
        "errors": report["errors"],
        "repair": repair_plan(manifest),
        "qa": qa,
        "readyJobs": [item["id"] for item in ready_jobs(manifest)],
    }


def _identity_error(errors: list[str]) -> bool:
    return any(
        "identity" in error or "palette overlap" in error or "color similarity" in error
        for error in errors
    )


def repair_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    """Choose the smallest failing scope: frames, then clip, then canonical base."""

    jobs = job_map(manifest)
    items: list[dict[str, Any]] = []
    identity_failures = 0
    inspected_clips = 0
    for job in manifest["jobs"]:
        if job.get("status") == "pending":
            continue
        inspection = job.get("inspection") or {}
        errors = list(inspection.get("errors") or [])
        if job.get("kind") == "clip-sheet":
            inspected_clips += 1
            if _identity_error(errors):
                identity_failures += 1
        failed_frames = list(inspection.get("failedFrames") or [])
        if job.get("status") != "failed" and not errors:
            continue
        scope = "clip"
        if job.get("kind") == "canonical-base":
            scope = "base"
        elif failed_frames and len(failed_frames) <= max(1, int(job.get("frame_count", 1)) // 3):
            scope = "frames"
        elif any("layout-guide" in error for error in errors):
            scope = "clip"
        items.append(
            {
                "job": job["id"],
                "scope": scope,
                "frames": failed_frames,
                "errors": errors,
            }
        )
    if items and (
        jobs["base"].get("status") != "complete"
        or (inspected_clips > 0 and identity_failures >= math.ceil(inspected_clips / 2))
    ):
        return {
            "scope": "base",
            "jobs": items,
            "note": "identity failed across the run; relock the canonical base before repairing clips",
        }
    if not items:
        return {"scope": "none", "jobs": [], "note": "no failed jobs"}
    if all(item["scope"] == "frames" for item in items) and sum(len(item["frames"]) for item in items) > 0:
        return {
            "scope": "frames",
            "jobs": items,
            "note": "replace only the listed frames, then re-accept those clips",
        }
    if len(items) == 1:
        return {
            "scope": items[0]["scope"],
            "jobs": items,
            "note": f"repair {items[0]['job']} at {items[0]['scope']} scope",
        }
    return {
        "scope": "clip",
        "jobs": items,
        "note": "repair each failed clip independently; do not regenerate the whole character",
    }


def clip_run_status(run_dir: str | Path) -> dict[str, Any]:
    manifest = load_clip_run(run_dir)
    return {
        "ok": True,
        "run": manifest["_root"],
        "character": manifest["character"],
        "canonicalBase": manifest["canonicalBase"],
        "canonicalBaseExists": Path(manifest["_root"], *_split_rel(manifest["canonicalBase"])).is_file(),
        "readyJobs": [job["id"] for job in ready_jobs(manifest)],
        "jobs": [
            {
                "id": job["id"],
                "kind": job["kind"],
                "status": job["status"],
                "depends_on": job["depends_on"],
                "failedFrames": (job.get("inspection") or {}).get("failedFrames", []),
            }
            for job in manifest["jobs"]
        ],
        "repair": repair_plan(manifest),
        "workerProtocol": manifest.get("workerProtocol"),
    }


def inspect_clip_run(run_dir: str | Path) -> dict[str, Any]:
    manifest = load_clip_run(run_dir)
    reports = []
    errors: list[str] = []
    for job in manifest["jobs"]:
        if job.get("status") == "pending":
            continue
        report = inspect_job(manifest, str(job["id"]))
        job["inspection"] = report
        job["status"] = "complete" if not report["errors"] else "failed"
        reports.append(report)
        errors.extend(report["errors"])
    qa = _refresh_qa(manifest)
    save_clip_run(manifest)
    _write_json(
        safe_output_child(_run_root(manifest["_root"]), QA_DIR, "review.json", label="qa review"),
        {
            **qa,
            "errors": errors,
            "jobs": reports,
        },
    )
    return {
        "ok": not errors,
        "errors": errors,
        "jobs": reports,
        "repair": repair_plan(manifest),
        "qa": qa,
        "readyJobs": [job["id"] for job in ready_jobs(manifest)],
    }


__all__ = [
    "accept_job",
    "clip_run_status",
    "create_layout_guide",
    "inspect_clip_run",
    "inspect_guide_pixels",
    "inspect_identity",
    "inspect_job",
    "load_clip_run",
    "prepare_clip_run",
    "ready_jobs",
    "repair_plan",
]

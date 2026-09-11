from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from assetforge.cli import main
from assetforge.clip_run import (
    GUIDE_MARK,
    accept_job,
    clip_run_status,
    inspect_guide_pixels,
    inspect_identity,
    load_clip_run,
    prepare_clip_run,
)
from assetforge.tests.test_release import _profile


def _blob(path: Path, color: tuple[int, int, int] = (180, 140, 110), size: int = 20, dx: int = 0) -> Path:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((4 + dx, 6, size - 5 + dx, size - 3), fill=(*color, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _sheet(
    path: Path,
    color: tuple[int, int, int] = (180, 140, 110),
    frames: int = 2,
    size: int = 20,
    *,
    vary: bool = True,
    y_shift: int = 0,
    scale: int = 1,
) -> Path:
    image = Image.new("RGBA", (size * frames, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for index in range(frames):
        left = index * size
        dx = index * 2 if vary else 0
        inset = 4 if scale == 1 else size // 2 - 1
        draw.rectangle(
            (left + inset + dx, 6 + y_shift, left + size - inset - 1 + dx, size - 3 + y_shift),
            fill=(*color, 255),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _marked_frame(path: Path, size: int = 20) -> Path:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 6, size - 5, size - 3), fill=(180, 140, 110, 255))
    draw.rectangle((1, 1, size - 2, 3), fill=(*GUIDE_MARK, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


class ClipRunTests(unittest.TestCase):
    def test_layout_guide_pixels_are_construction_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile(root)
            reference = _blob(root / "identity.png")
            prepared = prepare_clip_run(
                profile,
                character="demo",
                tier="runtime",
                direction="east",
                output=root / "run",
                reference=reference,
                lock_reference=True,
            )
            self.assertTrue(prepared["ok"], prepared)
            self.assertEqual(prepared["readyJobs"], ["idle"])
            guide = Image.open(root / "run" / "references" / "layout-guides" / "walk.png")
            report = inspect_guide_pixels(guide)
            self.assertTrue(report["errors"])
            jobs = {job["id"]: job for job in prepared["jobs"]}
            self.assertEqual(jobs["base"]["status"], "complete")
            self.assertEqual(jobs["walk"]["depends_on"], ["base", "idle"])

    def test_clip_jobs_wait_for_canonical_base_and_idle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile(root)
            prepared = prepare_clip_run(
                profile,
                character="demo",
                tier="runtime",
                direction="east",
                output=root / "run",
            )
            self.assertEqual(prepared["readyJobs"], ["base"])
            self.assertFalse(prepared["canonicalBaseExists"])
            manifest = load_clip_run(root / "run")
            with self.assertRaisesRegex(ValueError, "blocked by incomplete dependencies"):
                accept_job(manifest, "walk", source=_sheet(root / "walk.png"))

    def test_guide_pixels_and_identity_drift_are_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile(root)
            canonical = Image.open(_blob(root / "identity.png")).convert("RGBA")
            drifted = Image.open(_blob(root / "green.png", (16, 200, 48))).convert("RGBA")
            identity = inspect_identity(drifted, canonical, clip="walk")
            self.assertTrue(identity["errors"])
            marked = inspect_guide_pixels(Image.open(_marked_frame(root / "marked.png")))
            self.assertTrue(any("layout-guide" in error for error in marked["errors"]))

    def test_failed_frames_are_repaired_without_rescheduling_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile(root)
            prepared = prepare_clip_run(
                profile,
                character="demo",
                tier="runtime",
                direction="east",
                output=root / "run",
                reference=_blob(root / "identity.png"),
                lock_reference=True,
            )
            manifest = load_clip_run(prepared["run"])
            idle = accept_job(
                manifest,
                "idle",
                source=_sheet(root / "idle.png"),
                qa_note="idle matches the locked base",
            )
            self.assertTrue(idle["ok"], idle)
            self.assertEqual(idle["readyJobs"], ["walk"])
            walk = accept_job(manifest, "walk", source=_sheet(root / "walk.png"))
            self.assertTrue(walk["ok"], walk)
            failed = accept_job(
                manifest,
                "walk",
                source=_marked_frame(root / "walk-1.png"),
                frame=1,
            )
            self.assertFalse(failed["ok"], failed)
            self.assertEqual(failed["failedFrames"], [1])
            self.assertEqual(failed["repair"]["scope"], "frames")
            repaired = accept_job(
                manifest,
                "walk",
                source=_blob(root / "walk-1-fixed.png", dx=2),
                frame=1,
            )
            self.assertTrue(repaired["ok"], repaired)
            self.assertEqual(repaired["repair"]["scope"], "none")
            self.assertTrue((root / "run" / "qa" / "contact-sheet.png").is_file())

    def test_identity_failures_relock_the_canonical_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile(root)
            prepared = prepare_clip_run(
                profile,
                character="demo",
                tier="runtime",
                direction="east",
                output=root / "run",
                reference=_blob(root / "identity.png"),
                lock_reference=True,
            )
            manifest = load_clip_run(prepared["run"])
            self.assertTrue(accept_job(manifest, "idle", source=_sheet(root / "idle.png"))["ok"])
            drifted = accept_job(
                manifest,
                "walk",
                source=_sheet(root / "walk-green.png", (16, 200, 48)),
            )
            self.assertFalse(drifted["ok"], drifted)
            self.assertEqual(drifted["repair"]["scope"], "base")

    def test_force_replaces_an_existing_run_and_cli_reports_ready_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile(root)
            profile_path = root / "profile.json"
            identity = _blob(root / "identity.png")
            first = prepare_clip_run(
                profile,
                character="demo",
                tier="runtime",
                direction="east",
                output=root / "run",
                reference=identity,
                lock_reference=True,
            )
            self.assertTrue(first["ok"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare_clip_run(
                    profile,
                    character="demo",
                    tier="runtime",
                    direction="east",
                    output=root / "run",
                    reference=identity,
                    lock_reference=True,
                )
            second = prepare_clip_run(
                profile,
                character="demo",
                tier="runtime",
                direction="east",
                output=root / "run",
                reference=identity,
                lock_reference=True,
                force=True,
            )
            self.assertTrue(second["ok"], second)
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                code = main(
                    [
                        "clip-run",
                        "status",
                        "--run",
                        str(root / "run"),
                    ]
                )
            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["readyJobs"], ["idle"])
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                code = main(
                    [
                        "clip-run",
                        "prepare",
                        "--profile",
                        str(profile_path),
                        "--character",
                        "demo",
                        "--tier",
                        "runtime",
                        "--direction",
                        "east",
                        "--output",
                        str(root / "cli-run"),
                        "--reference",
                        str(identity),
                        "--lock-reference",
                    ]
                )
            self.assertEqual(code, 0, stdout.getvalue())
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["ok"], payload)
            self.assertEqual(payload["readyJobs"], ["idle"])
            status = clip_run_status(root / "cli-run")
            self.assertEqual(status["readyJobs"], ["idle"])

    def test_duplicate_walk_holds_fail_before_ingest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile(root)
            prepared = prepare_clip_run(
                profile,
                character="demo",
                tier="runtime",
                direction="east",
                output=root / "run",
                reference=_blob(root / "identity.png"),
                lock_reference=True,
            )
            manifest = load_clip_run(prepared["run"])
            self.assertTrue(accept_job(manifest, "idle", source=_sheet(root / "idle.png"))["ok"])
            cloned = accept_job(
                manifest,
                "walk",
                source=_sheet(root / "walk-clone.png", vary=False),
            )
            self.assertFalse(cloned["ok"], cloned)
            self.assertTrue(any("duplicate holds" in error for error in cloned["errors"]))

    def test_missing_mass_and_foot_drift_fail_identity_and_loop_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile(root)
            prepared = prepare_clip_run(
                profile,
                character="demo",
                tier="runtime",
                direction="east",
                output=root / "run",
                reference=_blob(root / "identity.png"),
                lock_reference=True,
            )
            tiny = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
            draw = ImageDraw.Draw(tiny)
            draw.rectangle((8, 8, 10, 10), fill=(180, 140, 110, 255))
            identity = inspect_identity(tiny, Image.open(root / "identity.png").convert("RGBA"), clip="walk")
            self.assertTrue(any("occupancy" in error or "bbox ratio" in error for error in identity["errors"]), identity)
            manifest = load_clip_run(prepared["run"])
            self.assertTrue(accept_job(manifest, "idle", source=_sheet(root / "idle.png"))["ok"])
            floating = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
            draw = ImageDraw.Draw(floating)
            draw.rectangle((4, 6, 15, 17), fill=(180, 140, 110, 255))
            draw.rectangle((26, 1, 37, 10), fill=(180, 140, 110, 255))
            float_path = root / "walk-float.png"
            floating.save(float_path)
            drifted = accept_job(manifest, "walk", source=float_path)
            self.assertFalse(drifted["ok"], drifted)
            self.assertTrue(any("foot-line" in error for error in drifted["errors"]), drifted)



if __name__ == "__main__":
    unittest.main()

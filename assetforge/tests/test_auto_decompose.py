from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from assetforge.auto_decompose import auto_decompose


class AutoDecomposeTests(unittest.TestCase):
    def test_chooses_reference_and_writes_automatic_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "reference").mkdir()
            image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            ImageDraw.Draw(image).rectangle((20, 12, 44, 58), fill=(120, 40, 80, 255))
            image.save(root / "reference/east.png")
            image.save(root / "walk_00.png")
            result = auto_decompose(
                root,
                root / "out",
                archetype="biped-side",
                character="test",
                height=96,
                clips=["idle"],
                resample="nearest",
            )
            self.assertTrue(result["ok"])
            report = json.loads(Path(result["report"]).read_text())
            self.assertEqual(report["frameCount"], 2)
            self.assertFalse(report["automaticGate"]["production"])
            self.assertTrue(Path(result["rig"]).is_file())

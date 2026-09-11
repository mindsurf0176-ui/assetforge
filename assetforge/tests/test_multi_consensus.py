import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from assetforge.multi_consensus import multi_consensus


class MultiConsensusTests(unittest.TestCase):
    def test_writes_motion_heatmap_and_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "frames"
            root.mkdir()
            for index, offset in enumerate((0, 2, 4, 2)):
                image = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
                draw = ImageDraw.Draw(image)
                draw.rectangle((16, 10, 30, 40), fill=(120, 40, 80, 255))
                draw.rectangle((30 + offset, 20, 36 + offset, 26), fill=(240, 220, 80, 255))
                image.save(root / f"frame_{index}.png")
            result = multi_consensus(root, Path(temporary) / "out", change_threshold=10, min_component=2)
            self.assertTrue(result["ok"])
            self.assertTrue(Path(result["heatmap"]).is_file())
            self.assertGreaterEqual(result["candidateCount"], 1)

    def test_can_limit_analysis_to_one_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "frames"
            root.mkdir()
            for clip in ("idle", "walk"):
                for index, offset in enumerate((0, 2, 4)):
                    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
                    ImageDraw.Draw(image).rectangle((10 + offset, 8, 18 + offset, 24), fill=(255, 255, 255, 255))
                    image.save(root / f"{clip}_{index}.png")
            result = multi_consensus(root, Path(temporary) / "walk-out", clip="walk", min_component=2)
            self.assertEqual(result["frameCount"], 3)

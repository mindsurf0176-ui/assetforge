import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from assetforge.vision_import import stage_vision_layers


class VisionImportTests(unittest.TestCase):
    def test_stages_manifest_layers_as_named_parts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layers = root / "layers"
            layers.mkdir()
            Image.new("RGBA", (12, 12), (255, 0, 0, 255)).save(layers / "layer-head.png")
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({"head": "layer-head.png"}), encoding="utf-8")
            result = stage_vision_layers(layers, mapping, root / "build/vision-parts", archetype="biped-side")
            self.assertEqual(result["layers"], ["head"])
            self.assertTrue((root / "build/vision-parts/head.png").is_file())

    def test_rejects_layer_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layers = root / "layers"
            layers.mkdir()
            Image.new("RGBA", (12, 12), (255, 0, 0, 255)).save(root / "outside.png")
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({"head": "../outside.png"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                stage_vision_layers(layers, mapping, root / "build/vision-parts", archetype="biped-side")

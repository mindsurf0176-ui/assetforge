from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from assetforge.frames import (
    harden_alpha,
    infer_source_sheet_anchors,
    remove_chroma_background,
    remove_checkerboard_background,
    remove_sheet_separator_lines,
    split_source_sheet,
)


class SourceSheetTests(unittest.TestCase):
    def test_split_source_sheet_pads_remainder_to_one_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sheet = Image.new("RGBA", (10, 4), (236, 244, 241, 255))
            draw = ImageDraw.Draw(sheet)
            for index, color in enumerate(((200, 50, 50, 255), (50, 200, 50, 255), (50, 50, 200, 255))):
                draw.rectangle((index * 3 + 1, 1, index * 3 + 2, 2), fill=color)
            sheet_path = root / "sheet.png"
            sheet.save(sheet_path)

            paths = split_source_sheet(sheet_path, root / "frames", columns=3, crop_height=4, prefix="attack")

            self.assertEqual(len(paths), 3)
            sizes = []
            for path in paths:
                with Image.open(path) as opened:
                    sizes.append(opened.size)
            self.assertEqual(sizes, [(4, 4)] * 3)

    def test_split_source_sheet_can_ignore_empty_trailing_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sheet = Image.new("RGBA", (8, 8), (0, 255, 0, 255))
            draw = ImageDraw.Draw(sheet)
            for index in range(3):
                x = (index % 2) * 4 + 1
                y = (index // 2) * 4 + 1
                draw.rectangle((x, y, x + 1, y + 1), fill=(220, 40, 40, 255))
            sheet_path = root / "sheet.png"
            sheet.save(sheet_path)

            paths = split_source_sheet(
                sheet_path,
                root / "frames",
                columns=2,
                rows=2,
                frame_count=3,
                prefix="attack",
            )

            self.assertEqual(len(paths), 3)

    def test_infer_source_sheet_anchors_uses_each_foreground_bottom_center(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = root / "frames"
            frames.mkdir()
            first = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
            ImageDraw.Draw(first).rectangle((1, 2, 3, 6), fill=(220, 40, 40, 255))
            second = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
            ImageDraw.Draw(second).rectangle((3, 1, 6, 5), fill=(40, 220, 40, 255))
            first.save(frames / "walk_00.png")
            second.save(frames / "walk_01.png")

            anchors, bounds = infer_source_sheet_anchors(sorted(frames.glob("*.png")))

            self.assertEqual(anchors, [(2, 6), (4, 5)])
            self.assertEqual(bounds, (1, 1, 7, 7))

    def test_chroma_background_removes_enclosed_key_color(self) -> None:
        image = Image.new("RGBA", (5, 5), (0, 255, 0, 255))
        ImageDraw.Draw(image).rectangle((1, 1, 3, 3), fill=(40, 40, 40, 255))
        cleaned = remove_chroma_background(image)
        self.assertEqual(cleaned.getpixel((0, 0))[3], 0)
        self.assertEqual(cleaned.getpixel((2, 2))[3], 255)

    def test_separator_line_removes_only_wide_thin_rows(self) -> None:
        image = Image.new("RGBA", (10, 8), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.line((0, 4, 9, 4), fill=(20, 20, 20, 255), width=1)
        draw.rectangle((4, 1, 5, 3), fill=(220, 40, 40, 255))
        cleaned = remove_sheet_separator_lines(image)
        self.assertEqual(cleaned.getpixel((0, 4))[3], 0)
        self.assertEqual(cleaned.getpixel((4, 2))[3], 255)

    def test_harden_alpha_removes_semitransparent_halo(self) -> None:
        image = Image.new("RGBA", (3, 1), (10, 20, 30, 0))
        image.putpixel((1, 0), (10, 20, 30, 128))
        image.putpixel((2, 0), (10, 20, 30, 255))

        hardened = harden_alpha(image, 20)

        self.assertEqual([hardened.getpixel((x, 0))[3] for x in range(3)], [0, 255, 255])

    def test_checkerboard_background_is_keyed_including_enclosed_tiles(self) -> None:
        image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        pixels = image.load()
        for y in range(64):
            for x in range(64):
                even = ((x // 16) + (y // 16)) % 2 == 0
                pixels[x, y] = (253, 253, 253, 255) if even else (246, 246, 246, 255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 43, 43), fill=(40, 80, 180, 255))
        draw.rectangle((28, 28, 35, 35), fill=(253, 253, 253, 255))

        cleaned = remove_checkerboard_background(image)

        self.assertEqual(cleaned.getpixel((0, 0))[3], 0)
        self.assertEqual(cleaned.getpixel((16, 0))[3], 0)
        self.assertEqual(cleaned.getpixel((20, 20))[:3], (40, 80, 180))
        self.assertGreater(cleaned.getpixel((30, 30))[3], 20)

    def test_checkerboard_pass_ignores_uniform_green_key(self) -> None:
        image = Image.new("RGBA", (32, 32), (0, 255, 0, 255))
        ImageDraw.Draw(image).rectangle((8, 8, 23, 23), fill=(40, 80, 180, 255))

        cleaned = remove_checkerboard_background(image)

        self.assertEqual(cleaned.getpixel((0, 0)), (0, 255, 0, 255))
        self.assertEqual(cleaned.getpixel((10, 10)), (40, 80, 180, 255))

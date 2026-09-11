import unittest

from PIL import Image, ImageDraw

from assetforge.frames import (
    neutral_foreground_fringe_pixels,
    remove_neutral_edge_halo,
    remove_neutral_foreground_fringe,
    remove_light_edge_matte,
    darken_bright_outer_silhouette,
    clear_bright_connected_component,
)


class TransparentHaloTests(unittest.TestCase):
    def test_darkens_only_pale_outer_silhouette(self) -> None:
        image = Image.new("RGBA", (9, 9), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((2, 2, 6, 6), fill=(35, 42, 58, 255))
        draw.rectangle((2, 2, 6, 2), fill=(210, 212, 220, 255))
        draw.point((4, 4), fill=(220, 222, 230, 255))
        draw.point((2, 4), fill=(0, 232, 255, 255))

        cleaned = darken_bright_outer_silhouette(image)

        self.assertEqual(cleaned.getpixel((4, 2)), (35, 42, 58, 255))
        self.assertEqual(cleaned.getpixel((4, 4)), (220, 222, 230, 255))
        self.assertEqual(cleaned.getpixel((2, 4)), (0, 232, 255, 255))

    def test_clears_only_seeded_bright_component(self) -> None:
        image = Image.new("RGBA", (11, 11), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((2, 2, 4, 4), fill=(215, 216, 222, 255))
        draw.rectangle((7, 7, 8, 8), fill=(215, 216, 222, 255))

        cleaned = clear_bright_connected_component(image, (3, 3))

        self.assertEqual(cleaned.getpixel((3, 3))[3], 0)
        self.assertEqual(cleaned.getpixel((7, 7)), (215, 216, 222, 255))

    def test_removes_attached_neutral_foreground_fringe_layers(self) -> None:
        image = Image.new("RGBA", (13, 13), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((2, 2, 10, 10), fill=(221, 223, 224, 255))
        draw.rectangle((4, 4, 8, 8), fill=(45, 50, 58, 255))

        self.assertEqual(neutral_foreground_fringe_pixels(image, min_rgb=190, max_channel_spread=42), 32)
        cleaned = remove_neutral_foreground_fringe(
            image,
            min_rgb=190,
            max_channel_spread=42,
            max_layers=2,
        )

        self.assertEqual(
            neutral_foreground_fringe_pixels(
                cleaned,
                min_rgb=190,
                max_channel_spread=42,
            ),
            0,
        )
        self.assertEqual(cleaned.getpixel((6, 6)), (45, 50, 58, 255))

    def test_removes_border_connected_white_matte(self) -> None:
        image = Image.new("RGBA", (9, 9), (255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((2, 2, 6, 6), fill=(45, 50, 58, 255))
        draw.point((4, 4), fill=(250, 250, 250, 255))

        cleaned = remove_neutral_edge_halo(image)

        self.assertEqual(cleaned.getpixel((0, 0))[3], 0)
        self.assertEqual(cleaned.getpixel((1, 1))[3], 0)
        self.assertEqual(cleaned.getpixel((4, 4)), (250, 250, 250, 255))
        self.assertEqual(cleaned.getpixel((4, 3)), (45, 50, 58, 255))

    def test_removes_semitransparent_border_matte_before_alpha_hardening(self) -> None:
        image = Image.new("RGBA", (7, 7), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 6, 6), outline=(255, 255, 255, 96), width=1)
        draw.rectangle((2, 2, 4, 4), fill=(45, 50, 58, 255))

        cleaned = remove_neutral_edge_halo(image)

        self.assertEqual(cleaned.getpixel((0, 3))[3], 0)
        self.assertEqual(cleaned.getpixel((3, 3)), (45, 50, 58, 255))

    def test_removes_mid_gray_checkerboard_rim_without_eating_interior_white(self) -> None:
        image = Image.new("RGBA", (11, 11), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((2, 2, 8, 8), fill=(210, 210, 212, 255))
        draw.rectangle((4, 4, 6, 6), fill=(40, 70, 160, 255))
        draw.point((5, 5), fill=(248, 248, 248, 255))

        cleaned = remove_light_edge_matte(image)

        self.assertEqual(cleaned.getpixel((2, 5))[3], 0)
        self.assertEqual(cleaned.getpixel((5, 5)), (248, 248, 248, 255))
        self.assertEqual(cleaned.getpixel((5, 4)), (40, 70, 160, 255))

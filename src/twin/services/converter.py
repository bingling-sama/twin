"""Minecraft skin layout converter.

Expands legacy single-layer skins (width == 2 * height) to square double-layer
skins by copying the upper half and synthesizing mirrored left limbs in the lower half,
porting Eidolon src/converter.rs and src/assets.rs.
"""

from PIL import Image

# 64x32 right leg: source rects on the upper half in pixel coords (x0, y0, x1, y1)
RIGHT_LEG_OUTSIDE_RANGE = (0, 20, 4, 32)
RIGHT_LEG_TOP_FRONT_RANGE = (4, 16, 8, 32)
RIGHT_LEG_BOTTOM_RANGE = (8, 16, 12, 20)
RIGHT_LEG_INSIDE_RANGE = (8, 20, 12, 32)
RIGHT_LEG_BACK_RANGE = (12, 20, 16, 32)

# 64x64 double-layer layout: left leg destination rects on the lower half
LEFT_LEG_OUTSIDE_RANGE = (24, 52, 28, 64)
LEFT_LEG_TOP_FRONT_RANGE = (20, 48, 24, 64)
LEFT_LEG_BOTTOM_RANGE = (24, 48, 28, 52)
LEFT_LEG_INSIDE_RANGE = (16, 52, 20, 64)
LEFT_LEG_BACK_RANGE = (28, 52, 32, 64)

# Right arm source rects on the upper half
RIGHT_ARM_OUTSIDE_RANGE = (40, 20, 44, 32)
RIGHT_ARM_TOP_RANGE = (44, 16, 48, 20)
RIGHT_ARM_FRONT_RANGE = (44, 20, 48, 32)
RIGHT_ARM_BOTTOM_RANGE = (48, 16, 52, 20)
RIGHT_ARM_INSIDE_RANGE = (48, 20, 52, 32)
RIGHT_ARM_BACK_RANGE = (52, 20, 56, 32)

# Left arm destination rects on the lower half
LEFT_ARM_INSIDE_RANGE = (32, 52, 36, 64)
LEFT_ARM_TOP_RANGE = (36, 48, 40, 52)
LEFT_ARM_FRONT_RANGE = (36, 52, 40, 64)
LEFT_ARM_BOTTOM_RANGE = (40, 48, 44, 52)
LEFT_ARM_OUTSIDE_RANGE = (40, 52, 44, 64)
LEFT_ARM_BACK_RANGE = (44, 52, 48, 64)

REGION_PAIRS = (
    # Right leg parts -> left leg positions
    (RIGHT_LEG_OUTSIDE_RANGE, LEFT_LEG_OUTSIDE_RANGE),
    (RIGHT_LEG_TOP_FRONT_RANGE, LEFT_LEG_TOP_FRONT_RANGE),
    (RIGHT_LEG_BOTTOM_RANGE, LEFT_LEG_BOTTOM_RANGE),
    (RIGHT_LEG_INSIDE_RANGE, LEFT_LEG_INSIDE_RANGE),
    (RIGHT_LEG_BACK_RANGE, LEFT_LEG_BACK_RANGE),
    # Right arm parts -> left arm positions
    (RIGHT_ARM_OUTSIDE_RANGE, LEFT_ARM_OUTSIDE_RANGE),
    (RIGHT_ARM_TOP_RANGE, LEFT_ARM_TOP_RANGE),
    (RIGHT_ARM_FRONT_RANGE, LEFT_ARM_FRONT_RANGE),
    (RIGHT_ARM_BOTTOM_RANGE, LEFT_ARM_BOTTOM_RANGE),
    (RIGHT_ARM_INSIDE_RANGE, LEFT_ARM_INSIDE_RANGE),
    (RIGHT_ARM_BACK_RANGE, LEFT_ARM_BACK_RANGE),
)


def _scale_rect(rect: tuple[int, int, int, int], hd_ratio: float) -> tuple[int, int, int, int]:
    return (
        int(rect[0] * hd_ratio),
        int(rect[1] * hd_ratio),
        int(rect[2] * hd_ratio),
        int(rect[3] * hd_ratio),
    )


def single2double(image: Image.Image) -> Image.Image:
    """Expand a legacy single-layer skin (width == 2 * height) to a square double-layer atlas.

    Copies the upper half and synthesizes mirrored left limbs in the lower half.
    Raises ValueError if input image dimensions are not 2:1.
    """
    width, height = image.size
    if width != height * 2:
        raise ValueError(
            f"Input image is not a single-layer skin (width must be 2*height, got {width}x{height})"
        )

    rgba = image.convert("RGBA")
    hd_ratio = width / 64.0

    output = Image.new("RGBA", (width, width), (0, 0, 0, 0))
    output.paste(rgba, (0, 0))

    for src_rect, dst_rect in REGION_PAIRS:
        src = _scale_rect(src_rect, hd_ratio)
        dst = _scale_rect(dst_rect, hd_ratio)
        cropped = rgba.crop(src)
        flipped = cropped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        output.paste(flipped, (dst[0], dst[1]))

    return output

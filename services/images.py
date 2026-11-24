"""Turning an uploaded file into an upright RGB image, or a clear reason why not."""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
ORIENTATION_TAG = 0x0112


class ImageRejected(ValueError):
    """The upload isn't an image this service accepts. ``status`` is the HTTP code."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status, self.code, self.detail = status, code, detail


@dataclass
class DecodedImage:
    image: Image.Image  # RGB, upright (EXIF orientation applied)
    format: str
    width: int
    height: int
    exif_orientation: int  # 1 = already upright; 2-8 = was transposed


def decode_upload(data: bytes, max_pixels: int, min_side_px: int) -> DecodedImage:
    """Decode with the format sniffed from the bytes, never from the file name."""
    if not data:
        raise ImageRejected(422, "empty_file", "The uploaded file is empty.")
    try:
        probe = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise ImageRejected(415, "unsupported_type", "The file is not a readable image.") from None
    except Exception as exc:  # noqa: BLE001  malformed files fail in many ways inside Pillow
        raise ImageRejected(
            422,
            "corrupt_image",
            f"The file looks like an image but can't be read ({type(exc).__name__}).",
        ) from None
    if probe.format not in ALLOWED_FORMATS:
        raise ImageRejected(
            415,
            "unsupported_type",
            f"{probe.format} images aren't accepted; use JPEG, PNG or WebP.",
        )
    # The header gives the size before any pixels are decompressed, so a decompression
    # bomb is refused here, cheaply.
    w, h = probe.size
    if w * h > max_pixels:
        raise ImageRejected(
            413, "too_many_pixels", f"{w}×{h} is over the limit of {max_pixels:,} pixels."
        )
    if min(w, h) < min_side_px:
        raise ImageRejected(
            422, "too_small", f"{w}×{h} is too small; each side must be ≥ {min_side_px} px."
        )
    try:
        orientation = int(probe.getexif().get(ORIENTATION_TAG, 1) or 1)
        probe.load()  # truncated or corrupt data fails here
        upright = ImageOps.exif_transpose(probe)  # phone photos stored sideways
        rgb = upright.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ImageRejected(
            422, "corrupt_image", f"The image couldn't be decoded ({exc})."
        ) from None
    return DecodedImage(rgb, probe.format, rgb.width, rgb.height, orientation)

"""Images built in memory, and a stand-in for the model."""

import io

from PIL import Image

from app.attributes import ATTRIBUTES


def garment_image(width=256, height=320, colour=(200, 30, 30)) -> Image.Image:
    img = Image.new("RGB", (width, height), (240, 240, 240))
    img.paste(Image.new("RGB", (width // 2, height // 2), colour), (width // 4, height // 4))
    return img


def encode(img: Image.Image, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


class FakeModel:
    """Scores the candidates in list order (first = best), so no model is needed."""

    def __init__(self):
        self.seen = []

    def predict(self, image, top_k=3):
        import numpy as np

        from services.model import rank

        self.seen.append(image.size)
        return {
            name: rank(cands, -np.arange(len(cands), dtype=float), top_k)
            for name, cands in ATTRIBUTES.items()
        }

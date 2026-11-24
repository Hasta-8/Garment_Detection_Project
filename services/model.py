"""Fashion-CLIP zero-shot attribute scoring.

The January code ran the full model five times per request (image + one candidate list
each). Here every candidate prompt is encoded once at startup, so a request needs one
image forward pass; the scores are the same CLIP logits, softmaxed over each attribute's
candidates. A softmax over a fixed list is a relative score among those candidates, not
a calibrated probability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.attributes import ATTRIBUTES, Candidates


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def rank(candidates: Candidates, logits: np.ndarray, top_k: int) -> dict:
    """Top label, its score and the ranked alternatives (the top label first)."""
    scores = softmax(logits)
    order = np.argsort(-scores, kind="stable")
    alts = [{"label": candidates[i][0], "score": round(float(scores[i]), 4)} for i in order[:top_k]]
    return {"label": alts[0]["label"], "score": alts[0]["score"], "alternatives": alts}


class FashionClip:
    def __init__(
        self,
        model_name: str,
        revision: str,
        cache_dir: Path,
        attributes: dict[str, Candidates] = ATTRIBUTES,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(
            model_name, revision=revision, cache_dir=cache_dir
        )
        self.model = CLIPModel.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
        self.model.eval()
        self.attributes = attributes
        self.logit_scale = float(self.model.logit_scale.detach().exp())
        self.text = {}
        with torch.no_grad():
            for name, cands in attributes.items():
                tok = self.processor(text=[p for _, p in cands], return_tensors="pt", padding=True)
                t = self.model.get_text_features(**tok)
                self.text[name] = (t / t.norm(dim=-1, keepdim=True)).numpy()

    def image_embedding(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            px = self.processor(images=image, return_tensors="pt")
            v = self.model.get_image_features(**px)
        return (v / v.norm(dim=-1, keepdim=True)).numpy()[0]

    def predict(self, image: Image.Image, top_k: int = 3) -> dict[str, dict]:
        v = self.image_embedding(image)
        return {
            name: rank(cands, self.logit_scale * (self.text[name] @ v), top_k)
            for name, cands in self.attributes.items()
        }

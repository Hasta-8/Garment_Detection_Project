"""G4: prompt wording for gender, colour and category, on G3's sample and split.

    uv run --group eval python eval/prompts.py <dataset-dir>

Image embeddings are computed once (the same model and preprocessing as the service);
each prompt set only changes the text side. The set with the best tune accuracy is
adopted if it beats the current prompts by >= 0.05 accuracy on test (tracker, G4).
Writes eval/results/g4_prompts.json.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attributes import macro_f1  # noqa: E402

from app.attributes import ATTRIBUTES  # noqa: E402
from app.config import MODEL, MODEL_REVISION, get_settings  # noqa: E402
from services.model import FashionClip  # noqa: E402

LABELS = {a: [label for label, _ in c] for a, c in ATTRIBUTES.items()}
SETS = {
    "gender": {
        "A": ["male", "female", "unisex"],
        "B": ["men's clothing", "women's clothing", "unisex clothing"],
        "C": [
            "a photo of a garment for men",
            "a photo of a garment for women",
            "a photo of a unisex garment",
        ],
    },
    "colour": {
        "A": LABELS["colour"],
        "B": [f"a {c} garment" for c in LABELS["colour"]],
        "C": [f"a photo of {c} clothing" for c in LABELS["colour"]],
    },
    "category": {
        "A": LABELS["category"],
        "B": [f"a photo of a {c}" for c in LABELS["category"]],
    },
}
GAIN = 0.05


def main() -> None:
    settings = get_settings()
    torch.set_num_threads(settings.torch_threads)
    sample = json.loads((ROOT / "eval" / "results" / "g3_sample.json").read_text())
    by_id = {x["id"]: x for x in sample}
    model = FashionClip(MODEL, MODEL_REVISION, settings.models_dir)
    emb = {}
    for f in sorted(Path(sys.argv[1]).glob("data/*.parquet")):
        for r in pq.read_table(f, columns=["id", "image"]).to_pylist():
            if r["id"] in by_id:
                emb[r["id"]] = model.image_embedding(
                    Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                )
    ids = [i for i in by_id if i in emb]
    V = np.stack([emb[i] for i in ids])

    out = {"experiment": "G4", "images": len(ids), "attributes": {}}
    for attr, sets in SETS.items():
        truth = [by_id[i]["truth"][attr] for i in ids]
        split = [by_id[i]["split"] for i in ids]
        res = {}
        for name, prompts in sets.items():
            with torch.no_grad():
                tok = model.processor(text=prompts, return_tensors="pt", padding=True)
                t = model.model.get_text_features(**tok)
                T = (t / t.norm(dim=-1, keepdim=True)).numpy()
            pred = [LABELS[attr][k] for k in (V @ T.T).argmax(axis=1)]
            res[name] = {"prompts_example": prompts[:3]}
            for s in ("tune", "test"):
                pairs = [
                    (t_, p_)
                    for t_, p_, sp in zip(truth, pred, split, strict=True)
                    if sp == s and t_ is not None
                ]
                res[name][s] = {
                    "n": len(pairs),
                    "accuracy": round(float(np.mean([a == b for a, b in pairs])), 3),
                    "macro_f1": round(macro_f1([a for a, _ in pairs], [b for _, b in pairs]), 3),
                }
        best = max(sets, key=lambda n: (res[n]["tune"]["accuracy"], n == "A"))
        gain = res[best]["test"]["accuracy"] - res["A"]["test"]["accuracy"]
        out["attributes"][attr] = {
            "sets": res,
            "best_on_tune": best,
            "test_gain_vs_A": round(gain, 3),
            "adopt": bool(best != "A" and gain >= GAIN),
        }
    (ROOT / "eval" / "results" / "g4_prompts.json").write_text(json.dumps(out, indent=1))
    for attr, v in out["attributes"].items():
        print(
            attr,
            {
                n: (s["tune"]["accuracy"], s["test"]["accuracy"], s["test"]["macro_f1"])
                for n, s in v["sets"].items()
            },
            "best",
            v["best_on_tune"],
            "gain",
            v["test_gain_vs_A"],
            "adopt",
            v["adopt"],
        )


if __name__ == "__main__":
    main()

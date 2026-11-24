"""G3: accuracy of category, colour and gender on catalogue photos with known labels.

    uv run --group eval python eval/attributes.py <dataset-dir>   # holds data/*.parquet

The dataset is Myntra's catalogue (benitomartin/fashion-product-images-small-384x512):
photos at 384×512 with Myntra's own article type, base colour and gender. The label
mapping, sample and abstention rule were fixed before running (tracker, G3). Writes
eval/results/g3_attributes.json (metrics and product ids only; no images) and
eval/results/g3_sample.json (the sampled ids, their split and the model's answers).
"""

from __future__ import annotations

import io
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.attributes import ATTRIBUTES  # noqa: E402
from app.config import MODEL, MODEL_REVISION, get_settings  # noqa: E402

SEED = 20261124
PER_CATEGORY = 150
CATEGORY = {
    "Tshirts": "t-shirt",
    "Shirts": "shirt",
    "Jeans": "jeans",
    "Dresses": "dress",
    "Skirts": "skirt",
    "Jackets": "jacket",
    "Sweaters": "sweater",
    "Shorts": "shorts",
    "Sweatshirts": "sweatshirt",
    "Blazers": "blazer",
    "Trousers": "trousers",
    "Leggings": "leggings",
    "Kurtas": "kurta",
    "Sarees": "saree",
    "Suits": "suit",
    "Tops": "top",
    "Tracksuits": "tracksuit",
}
EXACT_COLOURS = [label for label, _ in ATTRIBUTES["colour"]]
COLOUR = {c.title(): c for c in EXACT_COLOURS} | {
    "Navy Blue": "navy",
    "Turquoise Blue": "turquoise",
    "Grey Melange": "grey",
    "Charcoal": "grey",
    "Off White": "white",
    "Coffee Brown": "brown",
    "Sea Green": "green",
    "Lime Green": "green",
    "Fluorescent Green": "green",
}
GENDER = {"Men": "men", "Women": "women", "Unisex": "unisex"}
TARGET_ACCURACY, MIN_COVERAGE = 0.90, 0.20


def macro_f1(truth: list[str], pred: list[str]) -> float:
    f1s = []
    for c in sorted(set(truth)):
        tp = sum(t == c and p == c for t, p in zip(truth, pred, strict=True))
        fp = sum(t != c and p == c for t, p in zip(truth, pred, strict=True))
        fn = sum(t == c and p != c for t, p in zip(truth, pred, strict=True))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def scores(items: list[dict]) -> dict:
    truth = [x["truth"] for x in items]
    pred = [x["label"] for x in items]
    per_class = defaultdict(lambda: [0, 0])
    for t, p in zip(truth, pred, strict=True):
        per_class[t][1] += 1
        per_class[t][0] += t == p
    return {
        "n": len(items),
        "accuracy": float(np.mean([t == p for t, p in zip(truth, pred, strict=True)]))
        if items
        else None,
        "macro_f1": macro_f1(truth, pred) if items else None,
        "recall_by_class": {
            c: {"n": n, "recall": round(k / n, 3)} for c, (k, n) in sorted(per_class.items())
        },
    }


def pick_threshold(items: list[dict]) -> float | None:
    """Smallest top-score threshold with >= 90 % accuracy on what is answered (tune)."""
    for t in sorted({x["score"] for x in items}):
        kept = [x for x in items if x["score"] >= t]
        if len(kept) / len(items) < MIN_COVERAGE:
            return None
        if np.mean([x["truth"] == x["label"] for x in kept]) >= TARGET_ACCURACY:
            return float(t)
    return None


def abstention(items: list[dict], t: float | None) -> dict:
    if t is None:
        return {"threshold": None, "coverage": 1.0, "accuracy_answered": scores(items)["accuracy"]}
    kept = [x for x in items if x["score"] >= t]
    return {
        "threshold": round(t, 4),
        "coverage": round(len(kept) / len(items), 3),
        "accuracy_answered": round(float(np.mean([x["truth"] == x["label"] for x in kept])), 3)
        if kept
        else None,
    }


def main() -> None:
    import torch

    from services.model import FashionClip

    settings = get_settings()
    torch.set_num_threads(settings.torch_threads)
    files = sorted(Path(sys.argv[1]).glob("data/*.parquet"))
    cols = ["id", "masterCategory", "articleType", "baseColour", "gender"]
    rows = [r for f in files for r in pq.read_table(f, columns=cols).to_pylist()]
    by_cat = defaultdict(list)
    for r in rows:
        if r["masterCategory"] == "Apparel" and r["articleType"] in CATEGORY:
            by_cat[r["articleType"]].append(r["id"])
    rng = random.Random(SEED)
    sample = []
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])
        sample += rng.sample(ids, min(PER_CATEGORY, len(ids)))
    rng.shuffle(sample)
    split = {i: ("tune" if k < len(sample) // 2 else "test") for k, i in enumerate(sample)}
    wanted = set(sample)

    model = FashionClip(MODEL, MODEL_REVISION, settings.models_dir)
    results, t0 = [], time.perf_counter()
    for f in files:
        for r in pq.read_table(f).to_pylist():
            if r["id"] not in wanted:
                continue
            image = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            pred = model.predict(image, top_k=3)
            results.append(
                {
                    "id": r["id"],
                    "split": split[r["id"]],
                    "truth": {
                        "category": CATEGORY[r["articleType"]],
                        "colour": COLOUR.get(r["baseColour"]),
                        "gender": GENDER.get(r["gender"]),
                        "source_colour": r["baseColour"],
                    },
                    "pred": {
                        k: {"label": v["label"], "score": v["score"]} for k, v in pred.items()
                    },
                }
            )
            if len(results) % 200 == 0:
                print(f"{len(results)}/{len(sample)} images", flush=True)
    secs = time.perf_counter() - t0

    report = {
        "experiment": "G3",
        "model": f"{MODEL}@{MODEL_REVISION[:7]}",
        "images": len(results),
        "seconds_per_image": round(secs / len(results), 3),
        "sample": {
            "per_category_cap": PER_CATEGORY,
            "seed": SEED,
            "category_counts": dict(Counter(r["truth"]["category"] for r in results)),
        },
        "attributes": {},
    }
    for attr in ("category", "colour", "gender"):
        items = {
            s: [
                {"truth": r["truth"][attr], **r["pred"][attr]}
                for r in results
                if r["split"] == s and r["truth"][attr] is not None
            ]
            for s in ("tune", "test")
        }
        t = pick_threshold(items["tune"])
        report["attributes"][attr] = {
            "test": scores(items["test"]),
            "all": {
                k: v
                for k, v in scores(items["tune"] + items["test"]).items()
                if k != "recall_by_class"
            },
            "abstention": {
                "rule": f"smallest threshold with >= {TARGET_ACCURACY} accuracy on tune; "
                f"not adopted below {MIN_COVERAGE} coverage",
                "tune": abstention(items["tune"], t),
                "test": abstention(items["test"], t),
            },
        }
    cats = sorted({r["truth"]["category"] for r in results})
    labels = [c for c, _ in ATTRIBUTES["category"]]
    cm = {
        t: dict(
            Counter(
                r["pred"]["category"]["label"]
                for r in results
                if r["split"] == "test" and r["truth"]["category"] == t
            )
        )
        for t in cats
    }
    report["category_confusion_test"] = {"rows_truth": cats, "columns_pred": labels, "counts": cm}
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    (out / "g3_attributes.json").write_text(json.dumps(report, indent=1))
    (out / "g3_sample.json").write_text(json.dumps(results, indent=0))
    summary = {
        a: {
            "test_accuracy": round(v["test"]["accuracy"], 3),
            "test_macro_f1": round(v["test"]["macro_f1"], 3),
            "n_test": v["test"]["n"],
            "abstention_test": v["abstention"]["test"],
        }
        for a, v in report["attributes"].items()
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

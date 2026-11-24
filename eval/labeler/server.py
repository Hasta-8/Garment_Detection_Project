"""Sleeve and pattern labeller for experiment G3 (local tool, not part of the service).

    uv run --group eval python eval/labeler/server.py <dataset-dir>
    → open http://127.0.0.1:8005

Shows 100 catalogue photos from G3's test half (tops, shirts, dresses, kurtas, …; no
bottoms or sarees), seeded. You pick a sleeve type and a pattern for each. No model
output is shown. Labels go to eval/labels/sleeve_pattern.json (git-ignored: the
photos can't be redistributed).
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
LABELS = EVAL / "labels" / "sleeve_pattern.json"
SEED, PER_CATEGORY, TOTAL = 20261125, 10, 100
WITH_SLEEVES = [
    "t-shirt",
    "shirt",
    "kurta",
    "top",
    "dress",
    "sweatshirt",
    "sweater",
    "jacket",
    "blazer",
    "tracksuit",
]
SLEEVES = ["sleeveless", "short", "three quarter", "long", "can't tell"]
sys.path.insert(0, str(EVAL.parent))
from app.attributes import ATTRIBUTES  # noqa: E402

PATTERNS = [label for label, _ in ATTRIBUTES["pattern"]] + ["other", "can't tell"]


class Label(BaseModel):
    sleeve: str
    pattern: str


def pick() -> list[int]:
    sample = json.loads((EVAL / "results" / "g3_sample.json").read_text())
    by_cat = defaultdict(list)
    for x in sample:
        if x["split"] == "test" and x["truth"]["category"] in WITH_SLEEVES:
            by_cat[x["truth"]["category"]].append(x["id"])
    rng = random.Random(SEED)
    chosen, rest = [], []
    for c in WITH_SLEEVES:
        ids = sorted(by_cat[c])
        rng.shuffle(ids)
        chosen += ids[:PER_CATEGORY]
        rest += ids[PER_CATEGORY:]
    rng.shuffle(rest)
    chosen += rest[: TOTAL - len(chosen)]  # small categories leave room: fill from the rest
    rng.shuffle(chosen)
    return chosen[:TOTAL]


def create_app(dataset: Path) -> FastAPI:
    order = pick()
    want = set(order)
    images = {}
    for f in sorted(dataset.glob("data/*.parquet")):
        for r in pq.read_table(f, columns=["id", "image"]).to_pylist():
            if r["id"] in want:
                images[r["id"]] = r["image"]["bytes"]
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    saved = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    app = FastAPI(title="Sleeve and pattern labeller")

    @app.get("/")
    def index():
        return FileResponse(HERE / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get("/api/items")
    def items():
        return {"order": order, "labels": saved, "sleeves": SLEEVES, "patterns": PATTERNS}

    @app.get("/api/image/{pid}")
    def image(pid: int):
        if pid not in images:
            raise HTTPException(404, "unknown item")
        return Response(images[pid], media_type="image/jpeg")

    @app.post("/api/label/{pid}")
    def label(pid: int, body: Label):
        if pid not in images:
            raise HTTPException(404, "unknown item")
        if body.sleeve not in SLEEVES or body.pattern not in PATTERNS:
            raise HTTPException(422, "unknown sleeve or pattern")
        saved[str(pid)] = body.model_dump()
        LABELS.write_text(json.dumps(saved, indent=1))
        return {"saved": len(saved)}

    return app


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    uvicorn.run(create_app(Path(sys.argv[1])), host="127.0.0.1", port=8005)

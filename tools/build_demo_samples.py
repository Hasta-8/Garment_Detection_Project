"""Build the demo's samples from sample_images/: a resized copy of each photo and the
API's response for it (precomputed through the real /v1/analyze, in-process).

    uv run python tools/build_demo_samples.py
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402

SRC = ROOT / "sample_images"
OUT = ROOT / "demo" / "samples"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    credits = json.loads((SRC / "credits.json").read_text())
    with TestClient(create_app(get_settings())) as client:
        while client.get("/ready").status_code != 200:
            time.sleep(0.3)
        manifest = []
        for c in credits:
            sid = Path(c["file"]).stem
            img = Image.open(SRC / c["file"]).convert("RGB")
            img.thumbnail((900, 900))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=88)
            (OUT / f"{sid}.jpg").write_bytes(buf.getvalue())
            # Precomputed from the same resized copy the page shows (and re-sends on "Run live").
            r = client.post(
                "/v1/analyze", files={"file": (f"{sid}.jpg", buf.getvalue(), "image/jpeg")}
            )
            r.raise_for_status()
            body = r.json() | {"request_id": "precomputed"}
            (OUT / f"{sid}.json").write_text(json.dumps(body, indent=1))
            manifest.append(
                {
                    "id": sid,
                    "image": f"samples/{sid}.jpg",
                    "result": f"samples/{sid}.json",
                    "credit": {
                        "title": c["title"],
                        "page": c["page"],
                        "author": c["author"],
                        "licence": c["licence"],
                        "licence_url": c["licence_url"],
                    },
                }
            )
            print(sid, "|", body["title"])
        (OUT / "samples.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

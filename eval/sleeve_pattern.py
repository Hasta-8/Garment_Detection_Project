"""G3 (sleeve, pattern): the model's answers vs the user's labels on 100 test-half photos.

    uv run --group eval python eval/sleeve_pattern.py

Predictions are the ones G3 recorded (eval/results/g3_sample.json; sleeve and pattern
prompts are unchanged since). "can't tell" labels are left out. Writes
eval/results/g3_sleeve_pattern.json (counts and metrics only).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from attributes import scores  # noqa: E402


def main() -> None:
    labels = json.loads((HERE / "labels" / "sleeve_pattern.json").read_text())
    preds = {
        str(x["id"]): x["pred"]
        for x in json.loads((HERE / "results" / "g3_sample.json").read_text())
    }
    out = {"experiment": "G3 sleeve/pattern", "labelled": len(labels)}
    for attr in ("sleeve", "pattern"):
        items = [
            {"truth": v[attr], "label": preds[k][attr]["label"], "score": preds[k][attr]["score"]}
            for k, v in labels.items()
            if v[attr] not in ("can't tell",)
        ]
        s = scores(items)
        out[attr] = s | {
            "truth_counts": dict(Counter(x["truth"] for x in items)),
            "confusions": dict(
                Counter(
                    f"{x['truth']} -> {x['label']}" for x in items if x["truth"] != x["label"]
                ).most_common(10)
            ),
        }
    (HERE / "results" / "g3_sleeve_pattern.json").write_text(json.dumps(out, indent=1))
    for attr in ("sleeve", "pattern"):
        o = out[attr]
        print(
            attr,
            "n",
            o["n"],
            "accuracy",
            round(o["accuracy"], 3),
            "macro-F1",
            round(o["macro_f1"], 3),
        )
        print("   recall", {k: (v["recall"], v["n"]) for k, v in o["recall_by_class"].items()})
        print("   top confusions", o["confusions"])


if __name__ == "__main__":
    main()

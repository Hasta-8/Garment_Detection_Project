import numpy as np
import pytest
from helpers import encode, garment_image

from app.attributes import ATTRIBUTES, describe
from services.images import ImageRejected, decode_upload
from services.model import rank, softmax


def test_softmax_is_stable_and_sums_to_one():
    s = softmax(np.array([1000.0, 1001.0, 999.0]))
    assert np.isclose(s.sum(), 1) and s.argmax() == 1


def test_rank_orders_candidates_and_keeps_top_k():
    cands = [("a", "pa"), ("b", "pb"), ("c", "pc"), ("d", "pd")]
    r = rank(cands, np.array([0.1, 3.0, 2.0, -1.0]), top_k=2)
    assert r["label"] == "b" and [a["label"] for a in r["alternatives"]] == ["b", "c"]
    assert r["score"] == r["alternatives"][0]["score"] > r["alternatives"][1]["score"]


def test_rank_ties_keep_list_order():
    r = rank([("x", "x"), ("y", "y")], np.array([1.0, 1.0]), top_k=2)
    assert r["label"] == "x" and r["score"] == 0.5


def test_candidate_lists_are_well_formed():
    assert set(ATTRIBUTES) == {"category", "colour", "pattern", "sleeve", "gender"}
    for name, cands in ATTRIBUTES.items():
        labels = [label for label, _ in cands]
        assert len(labels) == len(set(labels)) >= 3, name
        assert all(label and prompt for label, prompt in cands)


def test_describe_uses_a_fixed_template():
    title, desc = describe(
        {
            "category": "shirt",
            "colour": "blue",
            "pattern": "checked",
            "sleeve": "long",
            "gender": "men",
        }
    )
    assert title == "Men's blue checked shirt, long sleeves"
    assert "Fashion-CLIP" in desc
    t2, _ = describe(
        {
            "category": "dress",
            "colour": "red",
            "pattern": "plain",
            "sleeve": "sleeveless",
            "gender": "women",
        }
    )
    assert t2 == "Women's red plain dress, sleeveless"


def test_decode_accepts_common_formats():
    for fmt in ("JPEG", "PNG", "WEBP"):
        d = decode_upload(encode(garment_image(), fmt), 10**8, 32)
        assert d.format == fmt and d.image.mode == "RGB"


@pytest.mark.parametrize(
    ("data", "status", "code"),
    [
        (b"", 422, "empty_file"),
        (b"hello", 415, "unsupported_type"),
        (encode(garment_image())[:200], 422, "corrupt_image"),
    ],
)
def test_decode_rejects_bad_uploads(data, status, code):
    with pytest.raises(ImageRejected) as e:
        decode_upload(data, 10**8, 32)
    assert (e.value.status, e.value.code) == (status, code)


def test_decode_limits():
    with pytest.raises(ImageRejected) as e:
        decode_upload(encode(garment_image(2000, 2000)), 1_000_000, 32)
    assert e.value.code == "too_many_pixels"
    with pytest.raises(ImageRejected) as e:
        decode_upload(encode(garment_image(20, 20)), 10**8, 32)
    assert e.value.code == "too_small"

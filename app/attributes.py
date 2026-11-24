"""The attributes and their candidate labels: what the zero-shot classifier chooses from.

Each candidate has the label returned by the API and the text prompt given to
Fashion-CLIP. The prompts are the January 2026 ones, unchanged (experiment G2 checks the
refactor gives the same answers); where a prompt differs from its label, the label is the
clearer API value (e.g. prompt "Short Sleeves Garment" -> label "short").

`gender` is the catalogue section a garment is styled for (men's, women's, unisex), as
inferred by the model from the image: a product label, not a fact about any person.
"""

from __future__ import annotations

Candidates = list[tuple[str, str]]  # (label, prompt)


def same(*labels: str) -> Candidates:
    return [(label, label) for label in labels]


ATTRIBUTES: dict[str, Candidates] = {
    "category": same(
        "t-shirt",
        "shirt",
        "jeans",
        "dress",
        "skirt",
        "jacket",
        "sweater",
        "shorts",
        "hoodie",
        "blazer",
        "coat",
        "trousers",
        "leggings",
        "kurta",
        "saree",
        "suit",
        "top",
        "tank top",
        "tracksuit",
        "sweatshirt",
    ),
    "colour": same(
        "black",
        "white",
        "grey",
        "navy",
        "blue",
        "red",
        "pink",
        "purple",
        "green",
        "olive",
        "yellow",
        "orange",
        "beige",
        "brown",
        "tan",
        "maroon",
        "teal",
        "turquoise",
        "gold",
        "silver",
    ),
    "pattern": same(
        "striped",
        "checked",
        "plain",
        "floral",
        "polka dot",
        "abstract",
        "geometric",
        "animal print",
        "paisley",
        "tie-dye",
        "camouflage",
        "herringbone",
        "houndstooth",
        "ikat",
        "chevron",
        "argyle",
        "brocade",
        "lace",
        "embroidery",
        "graphic print",
    ),
    "sleeve": [
        ("sleeveless", "Sleeveless Garment"),
        ("short", "Short Sleeves Garment"),
        ("three quarter", "Three Quarter Sleeves Garment"),
        ("long", "Long Sleeves Garment"),
    ],
    # Prompts changed 2026-11-24 (experiment G4): the January ones ("male", "female",
    # "unisex") made the model answer "unisex" for almost everything (test accuracy 0.14);
    # these reach 0.94 on held-out catalogue photos.
    "gender": [
        ("men", "men's clothing"),
        ("women", "women's clothing"),
        ("unisex", "unisex clothing"),
    ],
}


def describe(labels: dict[str, str]) -> tuple[str, str]:
    """A product title and description assembled from the predicted labels by a fixed
    template (no language model writes them)."""
    who = {"men": "Men's", "women": "Women's", "unisex": "Unisex"}[labels["gender"]]
    c, col, pat, sl = labels["category"], labels["colour"], labels["pattern"], labels["sleeve"]
    sleeves = "sleeveless" if sl == "sleeveless" else f"{sl} sleeves"
    title = f"{who} {col} {pat} {c}, {sleeves}"
    description = (
        f"A {who.lower()} {c} in {col}, with a {pat} pattern and {sleeves}. "
        "Attributes inferred from the image by Fashion-CLIP."
    )
    return title, description

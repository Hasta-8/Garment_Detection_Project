# Garment attributes

Upload a photo of a garment and get structured catalogue attributes back: **category,
colour, pattern, sleeve type and gender section** (men's, women's, unisex), each with a
score and ranked alternatives, plus a product title assembled from them. It uses
[Fashion-CLIP](https://huggingface.co/patrickjohncyh/fashion-clip) zero-shot: the
image is compared with a text prompt for every candidate label, so no garment
classifier was trained. FastAPI service, CPU only.

> **Status:** complete. Hardened service, tests in CI, accuracy measured on held-out
> catalogue photos (below), Docker and a demo page. **Category, sleeve and gender
> section are usable; colour and especially pattern are not reliable.**

**▶ [Try the online demo](https://hasta-8.github.io/Garment_Detection_Project/)** (precomputed results, runs in the browser) · **[2-minute narrated walkthrough video](https://youtu.be/dFRQeSfDtRI)**

## How it works

```
garment photo
  → Fashion-CLIP image embedding (one forward pass)
  → cosine similarity with the embedding of every candidate prompt
      (prompts encoded once at startup: 20 categories, 20 colours, 20 patterns,
       4 sleeve types, 3 gender sections)
  → per attribute: softmax over its candidates → label, score, top-3 alternatives
  → title and description filled in from the labels by a fixed template
```

- **Scores are relative, not probabilities.** Each score is a softmax over that
  attribute's fixed candidate list: 0.9 means "much closer to this prompt than to the
  others in the list", not "90 % likely to be right". A garment that matches none of
  the candidates still gets a confident-looking top label.
- **Gender section** is the catalogue section a garment is styled for, inferred from
  the image. It is a product label, not a statement about any person.
- **The title and description are a template**, filled in from the predicted labels.
  No language model writes them.

| Model | Revision (pinned) | Licence |
|---|---|---|
| [`patrickjohncyh/fashion-clip`](https://huggingface.co/patrickjohncyh/fashion-clip) (CLIP ViT-B/32 fine-tuned on fashion product data) | `7e3ba62` | MIT |

## Status

| Capability | Status |
|---|---|
| Zero-shot attributes with scores and ranked alternatives | IMPLEMENTED |
| FastAPI service: typed schemas, one error shape, OpenAPI docs at `/docs` | IMPLEMENTED |
| Upload validation: format sniffed from the bytes (JPEG/PNG/WebP), size and pixel limits, corrupt files | IMPLEMENTED |
| Model loaded at startup; `/health` and `/ready` | IMPLEMENTED |
| One image pass per request (prompts encoded once at startup) | IMPLEMENTED (measured below) |
| Unit and API tests in CI (stand-in model), lint, lock-file check | IMPLEMENTED |
| Reproducible CPU-only setup: uv lock file, pinned model revision | IMPLEMENTED |
| Accuracy per attribute on held-out labelled photos; score threshold study | IMPLEMENTED (`eval/`) |
| An "uncertain" answer in the API | PLANNED (the measured category threshold, 0.88, is a starting point) |
| Docker (model mounted read-only), demo page, online static copy | IMPLEMENTED |
| Walkthrough video on YouTube | DONE ([narrated walkthrough](https://youtu.be/dFRQeSfDtRI); an earlier silent version is attached to the [v0.2.0 release](https://github.com/Hasta-8/Garment_Detection_Project/releases/tag/v0.2.0)) |

## Quick start

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). CPU only.

```bash
uv sync                                          # CPU-only torch, versions from uv.lock
export GARMENT_MODELS_DIR=~/models/hf            # where the model goes (default .cache/models)
uv run python -c "from huggingface_hub import snapshot_download as d; d('patrickjohncyh/fashion-clip', revision='7e3ba62ce16b379a1ab479346b66f192e76f51b7', cache_dir='$GARMENT_MODELS_DIR', allow_patterns=['*.json','*.txt','model.safetensors'])"
uv run python run.py                             # http://127.0.0.1:8003, docs at /docs
```

```bash
curl http://127.0.0.1:8003/ready                 # 503 while the model loads, then 200
curl -F "file=@sample_images/striped-shirt.jpg" http://127.0.0.1:8003/v1/analyze
```

Settings are environment variables with the `GARMENT_` prefix (or a `.env` file):
`MODELS_DIR`, `TOP_K` (alternatives per attribute, default 3), `TORCH_THREADS`
(default 4), `MAX_UPLOAD_MB` (10), `MAX_PIXELS` (40 million), `MIN_SIDE_PX` (32).
Tests need no model: `uv run pytest`. Demo page: http://127.0.0.1:8003 once it runs.

**Docker:** `docker compose up --build` (the model folder is mounted read-only from
`GARMENT_MODELS_HOST_DIR`, default `../ai_models/huggingface`). Ready in about 12 s; the
first request takes a few seconds longer while the model warms up.

## API

| Method and path | Purpose |
|---|---|
| `POST /v1/analyze` | Multipart form with a `file` field. Returns the JSON below |
| `GET /health`, `GET /ready` | Liveness; readiness (503 until the model is loaded) |
| `GET /docs` | Interactive OpenAPI documentation |
| `POST /analyze` | The January path, kept as a deprecated alias (the response is now richer) |

```json
{
 "request_id": "3f2a\u2026",
 "attributes": {
  "category": {
   "label": "shirt",
   "score": 0.9739,
   "alternatives": [
    {
     "label": "shirt",
     "score": 0.9739
    },
    {
     "label": "top",
     "score": 0.0221
    },
    {
     "label": "blazer",
     "score": 0.002
    }
   ]
  },
  "colour": {
   "label": "white",
   "score": 0.486,
   "alternatives": [
    "\u2026"
   ]
  },
  "pattern": {
   "label": "striped",
   "score": 0.9958,
   "alternatives": [
    "\u2026"
   ]
  },
  "sleeve": {
   "label": "short",
   "score": 0.6599,
   "alternatives": [
    "\u2026"
   ]
  },
  "gender": {
   "label": "women",
   "score": 0.549,
   "alternatives": [
    "\u2026"
   ]
  }
 },
 "title": "Women's white striped shirt, short sleeves",
 "description": "A women's shirt in white, with a striped pattern and short sleeves. Attributes inferred from the image by Fashion-CLIP.",
 "image": {
  "format": "JPEG",
  "width": 680,
  "height": 900,
  "exif_orientation": 1
 },
 "timings": {
  "decode_ms": 21.0,
  "model_ms": 62.0,
  "total_ms": 84.0
 },
 "model": "patrickjohncyh/fashion-clip@7e3ba62",
 "settings": {
  "top_k": 3,
  "torch_threads": 4
 }
}
```

**Errors** always have the shape `{"error", "detail", "request_id"}`: 413
`too_large` / `too_many_pixels`, 415 `unsupported_type` (checked from the bytes, not
the file name), 422 `corrupt_image` / `too_small` / `empty_file` / `invalid_request`,
503 `not_ready` / `busy`.

## Accuracy (measured)

On Myntra catalogue photos (384×512, [`benitomartin/fashion-product-images-small-384x512`](https://huggingface.co/datasets/benitomartin/fashion-product-images-small-384x512))
with Myntra's own labels, which Fashion-CLIP never trained on: up to 150 per category
(2,115 photos, 16 categories), split at random into halves; numbers are on the held-out
half. Sleeve and pattern have no catalogue labels, so I labelled 100 of the test photos
by hand (one labeller, no model output shown). The label mappings, the sample and the
decision rules were written down before running (`eval/`).

| Attribute | Accuracy | Macro-F1 | Test items |
|---|---|---|---|
| Category | **0.74** | 0.74 | 1,058 |
| Sleeve | **0.83** | 0.76 | 100 (hand-labelled) |
| Gender section | **0.94** | 0.66 | 977 ("unisex" is rare: 3) |
| Colour | **0.60** | 0.47 | 988 |
| Pattern | **0.57** | 0.39 | 100 (hand-labelled) |

- **Gender was broken and is fixed:** with the January prompts ("male", "female",
  "unisex") the model answered "unisex" for almost everything: **0.14** accuracy. The
  prompts "men's clothing" / "women's clothing" / "unisex clothing" give **0.94**. They
  were chosen on the other half of the data; rewording the colour and category
  prompts didn't help, so those stayed.
- **Category:** near-perfect on sarees, jeans, trousers, shirts, shorts and dresses;
  weak where the catalogue's categories overlap the candidate list: Myntra's
  "Sweatshirts" are mostly called *hoodie* (many are hoodies), its "Tops" are called
  tank top, shirt or t-shirt, and t-shirts are often called *shirt*.
- **Colour:** good for blue, green, maroon and purple; poor for black (called tan, navy
  or grey; these photos are on models, so skin and other clothes are in the frame),
  navy (called blue) and brown.
- **Pattern:** fine for plain and checked; never right for geometric prints (12 of 12
  called checked or argyle), 43 % for graphic prints. A 20-way list of fine textile
  terms (ikat, brocade, argyle…) invites near-miss answers.
- **Sleeve:** mistakes are neighbouring types (long ↔ three quarter, sleeveless ↔
  short). It is predicted for every garment, including trousers and skirts, where it
  means nothing.
- **When to trust it:** for category, keeping only answers with score ≥ 0.88 (threshold
  chosen on the other half) answers 60 % of photos with 89 % accuracy. For colour and
  gender no threshold reaches 90 %.

## Sample results

The nine sample photos (Wikimedia Commons), as the service labels them. Comments are mine, by eye.

| Photo | Predicted title | Comment | Photo credit |
|---|---|---|---|
| [striped-shirt](sample_images/striped-shirt.jpg) | Women's white striped shirt, short sleeves | Shirt and stripes right; sleeves are long (folded shirt); colour is multi-coloured | [Robert Sheie](https://commons.wikimedia.org/wiki/File:Paul_Smith_striped_shirt.jpg), CC BY 2.0 |
| [plaid-shirt](sample_images/plaid-shirt.jpg) | Men's tan checked shirt, long sleeves | Checked shirt right; mostly black and red, not tan | [Punker1999](https://commons.wikimedia.org/wiki/File:Plaid_IZOD_men%27s_shirt.jpg), CC0 |
| [patterned-sweater](sample_images/patterned-sweater.jpg) | Men's navy argyle sweater, long sleeves | Right; "argyle" is the nearest name on the list for a Nordic knit | [Menswear Market](https://commons.wikimedia.org/wiki/File:Dale_of_Norway_sweater.jpg), CC BY 2.0 |
| [khadi-kurta](sample_images/khadi-kurta.jpg) | Men's blue ikat kurta, three quarter sleeves | Right except pattern (a small woven check, not ikat) and sleeve length (long) | [Purva4u](https://commons.wikimedia.org/wiki/File:Blue_khadi_kurta.jpg), CC BY-SA 3.0 |
| [denim-jacket](sample_images/denim-jacket.jpg) | Women's blue embroidery jacket, long sleeves | Category, colour, sleeves right; "embroidery" is wrong (bleached denim) | [Etelä-Karjalan museo](https://commons.wikimedia.org/wiki/File:1980s_blue_denim_jacket,_Finland_%E2%80%93_01.jpg), CC BY 4.0 |
| [leather-jacket](sample_images/leather-jacket.jpg) | Men's maroon embroidery jacket, long sleeves | Varsity jacket: category and colour right; "embroidery" for the lettering | [Peachyeung316](https://commons.wikimedia.org/wiki/File:Cherry_Red_Leather_jacket_from_Italian_brand.jpg), CC BY-SA 4.0 |
| [hoodie-shop](sample_images/hoodie-shop.jpg) | Unisex maroon embroidery hoodie, long sleeves | Hoodie right; it is black, not maroon; the shop wall and price tag are in the frame | [Syced](https://commons.wikimedia.org/wiki/File:%22Vaccinated%22_hoodie_on_sale,_reference_to_COVID-19.jpg), CC0 |
| [green-skirt](sample_images/green-skirt.jpg) | Women's teal ikat skirt, three quarter sleeves | Skirt right; sleeve is meaningless for a skirt but always predicted | [Hunt Museum and Inés Rivilla López](https://commons.wikimedia.org/wiki/File:Green_Poplin_Skirt_by_Sybil_Connolly-_full_length_front.jpg), CC BY-SA 4.0 |
| [brown-trousers](sample_images/brown-trousers.jpg) | Men's brown plain trousers, three quarter sleeves | Right, except sleeve (meaningless for trousers) | [Levi Strauss & Co.](https://commons.wikimedia.org/wiki/File:Jeans_MET_1978.175.3_B.jpg), CC0 |

## Performance (measured)

On a 4-core Intel i5-10210U laptop CPU (4 threads):

| | |
|---|---|
| Model load | about 5 s from a warm disk cache (23 s cold) |
| Per image, model only | about 60 ms (January version: about 0.5 s) |
| Memory | about 0.65 GB at rest (Docker container) |

The January code ran the whole model five times per request, once per attribute,
re-encoding the image each time. Now every candidate prompt is encoded once at
startup and each request needs one image pass. On the January sample images this gave
the same labels for all five attributes on 9 of 9 images, with scores within 5×10⁻⁵,
about 8× faster (checked before the gender prompts were changed).

## Limitations

- **Measured on catalogue photos only** (one garment, studio light). Photos of
  people in the street, flat-lays on patterned backgrounds or several garments are
  harder; see the sample photos.
- **Sleeve is predicted for bottoms too** (trousers, skirts), where it means nothing.
- **Fixed candidate lists:** a garment outside them (a dhoti, a cardigan, a
  two-colour garment) is forced onto the nearest label. There is no "none of these"
  answer yet.
- **One garment per photo:** with several garments or a person in the frame, the
  embedding mixes them.
- Colour names overlap (navy/blue, maroon/red, tan/brown/beige); the nearest prompt
  wins even when two are equally right.

## Repository

```
app/          FastAPI app (main.py), settings (config.py), attribute lists and the title template (attributes.py)
services/     model.py (Fashion-CLIP scoring), images.py (upload decoding)
schemas/      API response models
tests/        unit and API tests (stand-in model, no download)
eval/         evaluation scripts, results (metrics only), labelling tool
demo/         the demo page and its precomputed samples
tools/        build_demo_samples.py
sample_images/  openly licensed garment photos (credits.json)
Dockerfile, docker-compose.yml   the container (model mounted at run time)
```

## Licence

Code: [MIT](LICENSE). Fashion-CLIP: MIT. Sample photos: from Wikimedia Commons, each
under its own licence, listed in [`sample_images/credits.json`](sample_images/credits.json)
(CC0, CC BY and CC BY-SA).

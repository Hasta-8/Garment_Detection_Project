from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import utils

app = FastAPI()

def analyze_image(image):
    category = utils.detect_category(image)
    base_color = utils.detect_base_color(image)
    pattern = utils.detect_pattern(image)
    sleeve = utils.detect_sleeve(image)
    gender = utils.detect_gender(image)

    if gender == "unisex":
        title = f"Unisex {base_color} {category} with {pattern} pattern and {sleeve} sleeves"
        description = f"This is a unisex {category}. Primarily {base_color} in color, featuring a {pattern} pattern and {sleeve} sleeves. Perfect for any occasion."
    elif gender == "male":
        title = f"Men's {base_color} {category} with {pattern} pattern and {sleeve} sleeves"
        description = f"This is a men's {category}. Primarily {base_color} in color, featuring a {pattern} pattern and {sleeve} sleeves. Perfect for any occasion."
    else:
        title = f"Women's {base_color} {category} with {pattern} pattern and {sleeve} sleeves"
        description = f"This is a women's {category}. Primarily {base_color} in color, featuring a {pattern} pattern and {sleeve} sleeves. Perfect for any occasion."


    return {
        "category": category,
        "base_color": base_color,
        "pattern": pattern,
        "sleeve": sleeve,
        "title": title,
        "description": description
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = analyze_image(image)

    return JSONResponse(content=result)

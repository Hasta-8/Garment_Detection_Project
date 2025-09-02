from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load model once when server starts
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

def detect_category(image: Image.Image):
    '''
    Takes a PIL image and detects the garment category
    using Fashion-CLIP
    '''
    # Garment Categories
    GARMENT_TYPES= [
        "t-shirt", "shirt", "jeans", "dress", "skirt",
        "jacket", "sweater", "shorts", "hoodie", "blazer",
        "coat", "trousers", "leggings", "kurta", "saree",
        "suit", "top", "tank top", "tracksuit", "sweatshirt"
    ]

    # Prepare inputs for Fashion-CLIP
    inputs = processor(
        text=GARMENT_TYPES,
        images=image,
        return_tensors="pt",
        padding=True    
    )

    # Run Model
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
    best_category = GARMENT_TYPES[probs.argmax()]

    print(f"Category: {best_category}\nConfidence: {float(probs.max())}")
    return best_category

def detect_base_color(image: Image.Image):
    """
    Detect the base color of the garment using Fashion-CLIP.
    """
    # Define a set of candidate fashion colors
    GARMENT_COLORS = [
        "black", "white", "grey", "navy", "blue",
        "red", "pink", "purple", "green", "olive",
        "yellow", "orange", "beige", "brown", "tan",
        "maroon", "teal", "turquoise", "gold", "silver"
    ]

    # Prepare inputs for Fashion-CLIP
    inputs = processor(
        text=GARMENT_COLORS,
        images=image,
        return_tensors="pt",
        padding=True
    )

    # Run model (no gradients needed)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    best_color = GARMENT_COLORS[probs.argmax()]

    print(f"Base color: {best_color}\nConfidence: {float(probs.max())}")
    return best_color

def detect_pattern(image: Image.Image):
    """
    Takes a PIL image and predicts the fabric pattern using Fashion-CLIP.
    """

    PATTERN_CATEGORIES = [
        "striped", "checked", "plain", "floral", "polka dot",
        "abstract", "geometric", "animal print", "paisley", "tie-dye",
        "camouflage", "herringbone", "houndstooth", "ikat", "chevron",
        "argyle", "brocade", "lace", "embroidery", "graphic print"
    ]

    # Prepare inputs for Fashion-CLIP
    inputs = processor(
        text=PATTERN_CATEGORIES,
        images=image,
        return_tensors="pt",
        padding=True
    )

    # Run model
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    best_pattern = PATTERN_CATEGORIES[probs.argmax()]

    print(f"Pattern: {best_pattern}\nConfidence: {float(probs.max())}")
    return best_pattern

def detect_sleeve(image: Image.Image):
    """
    Predicts the sleeve type using Fashion-CLIP zero-shot classification.
    """
    SLEEVE_TYPES = [
        "Sleeveless Garment",
        "Short Sleeves Garment",
        "Three Quarter Sleeves Garment",
        "Long Sleeves Garment"
    ]

    inputs = processor(
        text=SLEEVE_TYPES,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    best_sleeve = SLEEVE_TYPES[probs.argmax()]

    if best_sleeve == "Sleeveless Garment":
        best_sleeve = "no"
    elif best_sleeve == "Short Sleeves Garment":
        best_sleeve = "short"
    elif best_sleeve == "Three Quarter Sleeves Garment":
        best_sleeve = "three quarter"
    else:
        best_sleeve = "long"

    print(f"Sleeve: {best_sleeve}\n confidence: {float(probs.max())}")
    
    return best_sleeve

def detect_gender(image: Image.Image):
    """
    Predicts the gender of the garment using Fashion-CLIP zero-shot classification.
    """
    GENDER = ["male", "female", "unisex"]

    inputs = processor(
        text=GENDER,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    best_gender = GENDER[probs.argmax()]

    print(f"Gender: {best_gender}\n confidence: {float(probs.max())}")

    return best_gender

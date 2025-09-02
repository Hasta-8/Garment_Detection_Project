# 👕 Garment Detection API

An image-based API built with **FastAPI** that analyzes garment photos and predicts:

- 🏷️ **Garment Category** (t-shirt, jeans, dress, etc.)
- 🎨 **Base Color** (red, blue, black, etc.)
- 🧵 **Fabric Pattern** (striped, floral, plain, etc.)
- 👕 **Sleeve Type** (short, long, sleeveless, etc.)
- 📝 **Auto-generated Title & Description** for e-commerce use

## 🚀 Features

- Upload an image of a garment via API.
- Uses **Fashion-CLIP** (Contrastive Language–Image Pretraining) for zero-shot garment analysis.
- Extracts attributes like **category, base color, sleeve type, and pattern**.
- Returns structured JSON response.
- Ready to integrate into **e-commerce platforms**

## 📂 Project Structure
```bash
garment_detection_project/
│── main.py # FastAPI app & API routes
│── utils.py # Attribute detection functions
│── requirements.txt # Dependencies
│── README.md # Documentation (this file)
│── sample_images/ # Example garment images
```

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/garment-detection-api.git
cd garment-detection-api
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run FastAPI app
```bash
uvicorn main:app --reload
```
The API will be live at 👉 `http://127.0.0.1:8000`

## 🧪 API Usage

### Endpoint:

`POST /analyze`

### Parameters:

- `file` → Image of garment (form-data upload).

### Example (Postman / curl):
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
-F "file=@sample_images/tshirt.jpg"
```

### Example Response:
```python
{
  "category": "t-shirt",
  "base_color": "red",
  "pattern": "striped",
  "sleeve": "short sleeves",
  "title": "Red Striped T-Shirt - Short Sleeves",
  "description": "Trendy red striped t-shirt with comfortable short sleeves, perfect for casual wear."
}
```

## 🚀 Future Improvements

- ### Enhanced Title & Description Generation

    - Integrate BLIP-2
    (Bootstrapping Language-Image Pretraining) to generate more human-like and context-aware product titles and descriptions for e-commerce.

    - Example: Instead of "red t-shirt", BLIP-2 could generate "Stylish Red Cotton T-Shirt with Comfortable Fit".

- ### Model Expansion

    - Add more garment categories, sleeve types, patterns, and color options for broader coverage.

    - Fine-tune Fashion-CLIP or use specialized fashion models for higher accuracy.
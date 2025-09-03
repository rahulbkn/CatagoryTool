from flask import Flask, jsonify
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Initialize Flask
app = Flask(__name__)

# Load pre-trained MobileNet model once at startup
model = MobileNet(weights="imagenet")

# Your backend API with API key
BACKEND_URL = "https://telegram-backend-r80h.onrender.com/api/files"
API_KEY = "b7f8a3c9d2e1f0b6a4c5d8e9f1a2b3c4"


def fetch_wallpaper_urls():
    """Fetch wallpaper URLs from backend."""
    try:
        headers = {"x-api-key": API_KEY}
        response = requests.get(BACKEND_URL, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get("success") and "files" in data:
            urls = [file["directLink"] for file in data["files"] if "directLink" in file]
            logging.info(f"Fetched {len(urls)} wallpaper URLs")
            return urls
        else:
            logging.warning("No files found in API response")
            return []
    except Exception as e:
        logging.error(f"Error fetching wallpaper URLs: {e}")
        return []


def categorize_image(image_url):
    """Categorize a single image using MobileNet."""
    try:
        response = requests.get(image_url, timeout=20)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((224, 224))

        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        decoded = decode_predictions(preds, top=1)[0]
        category = decoded[0][1]  # e.g., "mountain"
        return category
    except Exception as e:
        logging.error(f"Error categorizing {image_url}: {e}")
        return "Unknown"


@app.route("/classify-all", methods=["GET"])
def classify_all():
    """Fetch all wallpapers and return categorized results."""
    urls = fetch_wallpaper_urls()
    categorized = []

    for url in urls:
        category = categorize_image(url)
        categorized.append({"url": url, "category": category})

    return jsonify({"success": True, "count": len(categorized), "data": categorized})


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Wallpaper categorizer API is running!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

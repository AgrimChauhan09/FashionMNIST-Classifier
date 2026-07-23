from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import logging

# Logging setup
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = tf.keras.models.load_model("../models/fashion_mnist_model.keras")
logging.info("Model loaded successfully at startup")

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


@app.get("/")
def home():
    return {"message": "Fashion MNIST API Running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    logging.info(f"Request received: filename={file.filename}, content_type={file.content_type}")

    image_bytes = await file.read()
    logging.info(f"Image size: {len(image_bytes)} bytes")

    if len(image_bytes) == 0:
        logging.error("Uploaded file is empty")
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("L")

    except UnidentifiedImageError:
        logging.error(f"Invalid image file uploaded: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Upload PNG/JPG image."
        )

    except Exception as e:
        logging.error(f"Unexpected error while opening image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    image = image.resize((28, 28))

    image_array = np.array(image).astype("float32") / 255.0

    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=-1)

    prediction = model.predict(image_array, verbose=0)

    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    logging.info(f"Prediction: {class_names[predicted_class]} | Confidence: {confidence*100:.2f}%")

    return {
        "prediction": class_names[predicted_class],
        "confidence": round(confidence * 100, 2)
    }
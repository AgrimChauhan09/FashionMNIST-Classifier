from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io

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

    print("=" * 60)
    print("Filename :", file.filename)
    print("Content-Type :", file.content_type)

    image_bytes = await file.read()

    print("Image Size :", len(image_bytes), "bytes")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("L")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Upload PNG/JPG image."
        )

    except Exception as e:
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

    return {
        "prediction": class_names[predicted_class],
        "confidence": round(confidence * 100, 2)
    }
# 👕 FashionMNIST Image Classifier

An end-to-end Deep Learning application that classifies clothing images from the FashionMNIST dataset using a Convolutional Neural Network (CNN). The project includes model training, a FastAPI backend for inference, and a Streamlit frontend for user interaction.

---

## 🚀 Features

- Train a CNN model on the FashionMNIST dataset
- Image preprocessing and normalization
- FastAPI REST API for inference
- Streamlit frontend
- Upload and preview images
- Predict clothing category
- Display confidence score
- Training & Validation Accuracy/Loss plots
- Confusion Matrix
- Classification Report

---

## 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Backend | FastAPI |
| Frontend | Streamlit |
| Visualization | Matplotlib |
| Image Processing | Pillow |
| Data Processing | NumPy |
| API Testing | Requests |

---

# 📂 Project Structure

```text
FashionMNIST-Classifier/
│
├── backend/                 # FastAPI API (prediction server)
│   ├── main.py
│   └── requirements.txt
│
├── frontend/                # Streamlit UI (image upload page)
│   ├── app.py
│   ├── Dockerfile           # Frontend Docker image
│   └── requirements.txt
│
├── models/                  # Trained CNN model (.keras file)
│   └── fashion_mnist_model.keras
│
├── training/                # Model training code (optional)
│   └── train.py
│
├── images/                  # Screenshots / training plots
│
├── .github/workflows/
│   └── ci.yml               # GitHub auto-check
│
├── Dockerfile               # Backend Docker image
├── docker-compose.yml       # Backend + Frontend
├── .dockerignore
└── README.md
```

---

# 📊 Dataset

The project uses the **FashionMNIST** dataset consisting of grayscale clothing images of size **28×28 pixels**.

### Classes

- T-shirt / Top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

---

# 🧠 Model Architecture

The CNN model consists of:

- Conv2D (32 Filters)
- MaxPooling2D
- Conv2D (64 Filters)
- MaxPooling2D
- Flatten
- Dense (128 Neurons)
- Dropout (0.3)
- Dense (10 Neurons, Softmax)

---

# ⚙ Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 15 |
| Batch Size | 64 |

---

# 📈 Model Performance

## Test Accuracy

**91%**

### Classification Summary

| Metric | Score |
|---------|------|
| Accuracy | 91% |
| Macro F1 Score | 0.91 |
| Weighted F1 Score | 0.91 |

---

# 📉 Training Results

The following visualizations are generated during training:

- Training Accuracy Curve
- Validation Accuracy Curve
- Training Loss Curve
- Validation Loss Curve
- Confusion Matrix
- Classification Report

---

# 🌐 Backend API

Framework: **FastAPI**

## Endpoint

```
POST /predict
```

### Input

Image File

### Response

```json
{
    "prediction": "Dress",
    "confidence": 95.42
}
```

---

# 🖥 Frontend

The frontend is developed using **Streamlit**.

Features include:

- Upload Image
- Image Preview
- Predict Button
- Display Predicted Class
- Display Confidence Score
- Error Handling

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/FashionMNIST-Classifier.git
```

Go to project directory

```bash
cd FashionMNIST-Classifier
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Run Backend

```bash
cd backend
uvicorn main:app --reload
```

Backend runs on

```
http://127.0.0.1:8000
```

---

# ▶ Run Frontend

```bash
cd frontend
streamlit run app.py
```

Frontend runs on

```
http://localhost:8501
```

---

# 📸 Screenshots

- Home Page
![SnapShots](https://github.com/AgrimChauhan09/FashionMNIST-Classifier/blob/main/images/home.png?raw=true)
- Image Upload
![SnapShots](https://github.com/AgrimChauhan09/FashionMNIST-Classifier/blob/main/images/image_upload.png?raw=true)
- Prediction Result
![SnapShots](https://github.com/AgrimChauhan09/FashionMNIST-Classifier/blob/main/images/prediction.png?raw=true)
- Accuracy Curve
![SnapShots](https://github.com/AgrimChauhan09/FashionMNIST-Classifier/blob/main/images/accuracy_curve.png?raw=true)
- Loss Curve
![SnapShots](https://github.com/AgrimChauhan09/FashionMNIST-Classifier/blob/main/images/loss_curve.png?raw=true)
- Confusion Matrix
![SnapShots](https://github.com/AgrimChauhan09/FashionMNIST-Classifier/blob/main/images/confusion_matrix.png?raw=true)


---

# 📦 Deployment
Frontend:
- Streamlit (https://fashionmnist-classifier-by-agrim.streamlit.app/)

Backend:
- Render


# 👨‍💻 Author
**Agrim Chauhan**

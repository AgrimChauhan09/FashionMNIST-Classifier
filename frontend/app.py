import streamlit as st
import requests
from PIL import Image
import os

st.set_page_config(
    page_title="FashionMNIST Classifier",
    page_icon="👕",
    layout="centered"
)

st.sidebar.title("📚 FashionMNIST Classes")

classes = [
    "👕 T-shirt / Top",
    "👖 Trouser",
    "🧥 Pullover",
    "👗 Dress",
    "🧥 Coat",
    "👡 Sandal",
    "👔 Shirt",
    "👟 Sneaker",
    "👜 Bag",
    "🥾 Ankle Boot"
]

for c in classes:
    st.sidebar.write(c)

st.sidebar.markdown("---")
st.sidebar.info("Upload a clothing image and click Predict.")

st.title("👕 FashionMNIST Image Classifier")

st.write(
    """
Upload a clothing image and let the trained TensorFlow model
predict its category.
"""
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:
        st.markdown("### Image Info")
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Type:** {uploaded_file.type}")
        st.write(f"**Size:** {round(uploaded_file.size/1024,2)} KB")

    st.markdown("---")

    if st.button("🚀 Predict", use_container_width=True):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        with st.spinner("Predicting..."):

            try:

                API_URL = os.environ.get("BACKEND_URL", "https://fashionmnist-classifier.onrender.com/predict")
                response = requests.post(
                API_URL,
                files=files,
                timeout=60
                )

                if response.status_code == 200:

                    result = response.json()

                    prediction = result["prediction"]
                    confidence = float(result["confidence"])

                    st.success("Prediction Completed ✅")

                    st.markdown("## 🎯 Prediction")

                    st.info(f"### {prediction}")

                    st.markdown("### Confidence")

                    st.progress(min(confidence / 100, 1.0))

                    st.metric(
                        label="Confidence Score",
                        value=f"{confidence:.2f}%"
                    )

                else:

                    st.error(response.text)

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to backend.\n\n"
                    "Make sure FastAPI server is running."
                )

            except Exception as e:

                st.error(str(e))

else:

    st.warning("Please upload an image.")

st.markdown("---")

st.caption("Made with ❤️ by Agrim Chauhan")

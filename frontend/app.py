import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="Fashion MNIST")

st.title("👕 Fashion MNIST Classifier")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    st.progress(confidence)

    st.image(image, width=250)

    if st.button("Predict"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type,
            )
        }

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files=files,
                timeout=30,
            )

            if response.status_code == 200:

                result = response.json()

                st.success(
                    f"Prediction : {result['prediction']}"
                )

                st.write(
                    f"Confidence : {result['confidence']} %"
                )

            else:

                st.error(response.text)
    

        except Exception as e:

            st.error(str(e))







import streamlit as st
from PIL import Image
import models.model_convnextsmall as model_convnextsmall
import models.model_densenet201 as model_densenet201   
import models.model_efficientnetv2s as model_efficientnetv2s
import models.model_inceptionv3 as model_inceptionv3   
import models.model_resnet50v2 as model_resnet50v2   
import models.model_xception as model_xception   

st.set_page_config(page_title="Skin Disease Prediction", layout="centered")
st.title("🩺 Skin Disease Classification from Image")
st.write("Upload a skin lesion image and get predictions from six different models.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((299, 299))
    st.image(resized_image, caption="Uploaded Image", use_container_width=False)

    st.subheader("Model Predictions")
    with st.spinner("Classifying..."):
        predictions = [
            {"Model": "ConvNeXtSmall", "Result": model_convnextsmall.predict_skin_disease(image)},
            {"Model": "DenseNet201", "Result": model_densenet201.predict_skin_disease(image)},
            {"Model": "EfficientNetV2S", "Result": model_efficientnetv2s.predict_skin_disease(image)},
            {"Model": "InceptionV3", "Result": model_inceptionv3.predict_skin_disease(image)},
            {"Model": "ResNet50V2", "Result": model_resnet50v2.predict_skin_disease(image)},
            {"Model": "Xception", "Result": model_xception.predict_skin_disease(image)},
        ]

        results_table = []
        for pred in predictions:
            result = pred["Result"]
            if result and len(result) == 2:
                results_table.append({
                    "Model": pred["Model"],
                    "Top-1 Class": result[0]["class"],
                    "Top-1 Confidence": f"{result[0]['probability']*100:.2f} %",
                    "Top-2 Class": result[1]["class"],
                    "Top-2 Confidence": f"{result[1]['probability']*100:.2f} %",
                })
            else:
                results_table.append({
                    "Model": pred["Model"],
                    "Top-1 Class": "Prediction Failed",
                    "Top-1 Confidence": "-",
                    "Top-2 Class": "-",
                    "Top-2 Confidence": "-"
                })

        st.table(results_table)

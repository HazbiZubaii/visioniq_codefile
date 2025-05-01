import streamlit as st
from PIL import Image
import torch
from tokenizer.tokenizer import get_tokenizer
from model.vqa_model import load_model

# Load the model and tokenizer
model = load_model()
processor = get_tokenizer()



st.title("🖼️ Visual Question Answering App")
st.markdown("### 👩‍💻 Created by Hasbi Fathima VP")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Text Input for Question
question = st.text_input("Ask a question about the image...")

if uploaded_file and question:
    image = Image.open(uploaded_file).convert("RGB")

    # Preprocess the image and question
    inputs = processor(image, question, return_tensors="pt")

    # Generate answer using the model
    with torch.no_grad():
        output = model.generate(**inputs)

    answer = processor.decode(output[0], skip_special_tokens=True)

    # Display Results
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Question:** {question}")
    st.write(f"**Answer:** {answer}")


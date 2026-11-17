
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page configuration
st.set_page_config(
    page_title="E-Commerce Sentiment Analysis",
    page_icon="🛒",
    layout="centered"
)

# Logo
st.image("logo.png", width=120)

# Title
st.title("🛒 E-Commerce Sentiment Analysis")

st.markdown("""
Analyze customer reviews using a **fine-tuned DistilBERT NLP model**.

This application predicts whether an e-commerce review is
**Positive 😊** or **Negative 😡**.
""")

st.divider()

# Load fine-tuned model
@st.cache_resource
def load_model():
    model_path = "sentiment_model"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.eval()

    return model, tokenizer


model, tokenizer = load_model()

# Review input
st.subheader("📝 Enter Customer Review")

user_input = st.text_area(
    "Review",
    height=150,
    placeholder="Example: The product quality is amazing and delivery was very fast."
)

# Example reviews
st.info("""
**Example Reviews**

✅ *This product is excellent. I loved it.*

❌ *Very poor quality. Completely disappointed.*
""")

# Analyze button
if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if user_input.strip():

        with st.spinner("Analyzing review..."):

            # Tokenize input
            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            # Prediction
            with torch.no_grad():
                outputs = model(**inputs)

            # Convert logits to probabilities
            probabilities = torch.softmax(outputs.logits, dim=1)

            # Get prediction
            prediction = torch.argmax(probabilities, dim=1).item()

            # Confidence
            confidence = probabilities[0][prediction].item() * 100

            negative_probability = probabilities[0][0].item()
            positive_probability = probabilities[0][1].item()

        # Labels
        labels = {
            0: "Negative 😡",
            1: "Positive 😊"
        }

        result = labels[prediction]

        # Display result
        st.divider()
        st.subheader("🎯 Prediction Result")

        if prediction == 1:
            st.success(f"### {result}")
        else:
            st.error(f"### {result}")

        # Confidence score
        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)

        # Probability comparison
        st.subheader("📊 Confidence Comparison")

        st.bar_chart({
            "Negative 😡": [negative_probability],
            "Positive 😊": [positive_probability]
        })

    else:
        st.warning("⚠️ Please enter a customer review.")

# Footer
st.divider()

st.caption(
    "Developed using Streamlit • Hugging Face Transformers • PyTorch • DistilBERT"
)

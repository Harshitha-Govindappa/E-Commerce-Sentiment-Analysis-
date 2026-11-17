# 🛒 E-Commerce Sentiment Analysis

## Overview

This project analyzes e-commerce customer reviews and predicts whether the sentiment is **Positive 😊** or **Negative 😡** using a pretrained **DistilBERT transformer model** from Hugging Face.

The application provides an interactive **Streamlit web interface** for real-time sentiment analysis and confidence visualization.

## 📸 Application Preview

![E-Commerce Sentiment Analysis Home Page](home.png)

## Features

* Real-time sentiment prediction
* Interactive Streamlit web interface
* Positive and Negative sentiment classification
* Confidence score visualization
* Pretrained DistilBERT NLP model
* User-friendly customer review input

## Technologies Used

* Python
* Streamlit
* PyTorch
* Hugging Face Transformers
* DistilBERT
* NLP

## Model

**Model:** `distilbert-base-uncased-finetuned-sst-2-english`

The pretrained DistilBERT model is used to classify customer reviews into Positive or Negative sentiment.

## Installation

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Project Workflow

1. Enter an e-commerce customer review.
2. The review is tokenized using the Hugging Face tokenizer.
3. The pretrained DistilBERT model analyzes the text.
4. The application predicts Positive or Negative sentiment.
5. The confidence score is displayed.
6. A confidence comparison chart is shown.

## Example

**Positive Review:**

> The product quality is amazing and delivery was very fast.

**Negative Review:**

> Very poor quality. Completely disappointed.

## Project Objective

To build an interactive NLP application that demonstrates **sentiment analysis using transformer-based models** and provides real-time predictions through Streamlit.

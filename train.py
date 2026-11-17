
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# -------------------------
# 1. Load Dataset
# -------------------------
df = pd.read_csv("ecommerce_reviews_1000.csv")

print("Dataset columns:")
print(df.columns.tolist())

# Keep required columns
df = df[["review_text", "rating"]]

# Remove missing values
df.dropna(inplace=True)

# -------------------------
# 2. Create Sentiment Labels
# -------------------------
def create_sentiment(rating):
    if rating <= 2:
        return 0       # Negative
    elif rating >= 4:
        return 1       # Positive
    else:
        return None    # Remove rating 3


df["label"] = df["rating"].apply(create_sentiment)

# Remove neutral reviews
df.dropna(subset=["label"], inplace=True)

df["label"] = df["label"].astype(int)

print("\nSentiment distribution:")
print(df["label"].value_counts())

# -------------------------
# 3. Train-Test Split
# -------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["review_text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print(f"\nTraining samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# -------------------------
# 4. Tokenizer
# -------------------------
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=128
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=128
)

# -------------------------
# 5. Dataset Class
# -------------------------
class SentimentDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)


train_dataset = SentimentDataset(
    train_encodings,
    train_labels
)

val_dataset = SentimentDataset(
    val_encodings,
    val_labels
)

# -------------------------
# 6. Load DistilBERT
# -------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={
        0: "Negative",
        1: "Positive"
    },
    label2id={
        "Negative": 0,
        "Positive": 1
    }
)

# -------------------------
# 7. Training Arguments
# -------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    report_to="none"
)

# -------------------------
# 8. Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# -------------------------
# 9. Train Model
# -------------------------
print("\n🚀 Starting model training...")

trainer.train()

# -------------------------
# 10. Save Model
# -------------------------
model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")

print("\n✅ Training complete!")
print("📁 Model saved in: sentiment_model")


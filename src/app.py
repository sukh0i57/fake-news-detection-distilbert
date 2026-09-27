import gradio as gr
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)


# -----------------------------------
# Load model
# -----------------------------------

MODEL_NAME = "sukh0i57/fake-news-distilbert"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)
model.eval()


# -----------------------------------
# Prediction function
# -----------------------------------

def predict_news(text):

    if not text or not text.strip():
        return "⚠️ Please enter some news text.", 0.0, 0.0

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)

    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    real_probability = probabilities[0].item() * 100
    fake_probability = probabilities[1].item() * 100

    if fake_probability > real_probability:
        prediction = "FAKE"
    else:
        prediction = "REAL"

    return (
        prediction,
        round(real_probability, 2),
        round(fake_probability, 2)
    )


# -----------------------------------
# Gradio interface
# -----------------------------------

demo = gr.Interface(

    fn=predict_news,

    inputs=gr.Textbox(
        lines=12,
        placeholder="Paste the news article here...",
        label="News Article"
    ),

    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="REAL Probability (%)"),
        gr.Number(label="FAKE Probability (%)")
    ],

    title="📰 Fake News Detector",

    description=(
        "DistilBERT-based fake news classification system. "
        "Enter a news article and click Submit."
    ),

    flagging_mode="never"
)


# -----------------------------------
# Launch application
# -----------------------------------

if __name__ == "__main__":
    demo.launch()

````markdown
# 📰 Fake News Detection using DistilBERT

A Transformer-based NLP project that classifies news articles as **REAL** or **FAKE** using **DistilBERT**. The project also compares the Transformer model with a traditional **TF-IDF + Logistic Regression** baseline.

---

## 📌 Overview

Fake news detection is an NLP classification problem where a model learns patterns from previously labeled news articles and predicts whether a new article belongs to the **REAL** or **FAKE** class.

This project implements and compares two approaches:

- **TF-IDF + Logistic Regression** — traditional machine learning baseline
- **DistilBERT** — Transformer-based deep learning model

The trained DistilBERT model is also deployed through an interactive **Gradio interface** for making predictions.

> **Important:** This project is a text classification system, not an independent fact-checking system. The model does not verify claims against external sources.

---

## 🎯 Objectives

- Build a fake news classification system using NLP.
- Train a Transformer model using DistilBERT.
- Build a traditional TF-IDF + Logistic Regression baseline.
- Compare both approaches using standard classification metrics.
- Analyze duplicate and near-duplicate articles.
- Save and reload the trained model.
- Deploy the model through a Gradio interface.

---

## 📊 Dataset

The original dataset contained:

| Dataset | Articles |
|---|---:|
| Fake.csv | 23,481 |
| True.csv | 21,417 |
| **Total** | **44,898** |

After removing exact duplicate articles:

**38,646 unique articles** remained.

### Labels

| Label | Meaning |
|---:|---|
| 0 | REAL |
| 1 | FAKE |

### Input

The model uses:

```text
Title + Article Text
````

The `subject` and `date` fields were excluded from the model input.

During exploratory analysis, `subject` categories showed a strong relationship with the class labels. They were therefore excluded to reduce the possibility of the model learning dataset-specific shortcuts.

---

## 🧠 Methodology

```text
                    Dataset
                       │
                       ▼
              Remove Duplicates
                       │
                       ▼
              Create Text Input
              Title + Article Text
                       │
                       ▼
             Stratified 80/20 Split
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      TF-IDF + Logistic       DistilBERT
         Regression           Transformer
             │                   │
             └─────────┬─────────┘
                       ▼
                   Evaluation
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
     Accuracy       Precision       Recall
                       │
                       ▼
                      F1
                       │
                       ▼
               Gradio Interface
```

---

## 🤖 DistilBERT Model

The primary model used in this project is:

```text
distilbert-base-uncased
```

### Configuration

| Parameter               |      Value |
| ----------------------- | ---------: |
| Model                   | DistilBERT |
| Maximum sequence length |        256 |
| Batch size              |         16 |
| Learning rate           |       2e-5 |
| Epochs                  |          2 |
| Optimizer               |      AdamW |
| Number of classes       |          2 |
| Parameters              | 66,955,010 |

### Tokenization

The DistilBERT tokenizer converts the article text into token IDs and attention masks.

```python
tokenizer(
    text,
    truncation=True,
    padding="max_length",
    max_length=256
)
```

---

## 📚 Train / Validation Split

A stratified 80/20 train-validation split was used.

| Dataset    | Articles |
| ---------- | -------: |
| Training   |   30,916 |
| Validation |    7,730 |
| Total      |   38,646 |

Stratification was used to preserve the class distribution between the training and validation sets.

---

# 📈 Results

The following results were obtained on the held-out validation split used in this experiment.

| Model                        |   Accuracy |  Precision |      Recall |   F1-Score |
| ---------------------------- | ---------: | ---------: | ----------: | ---------: |
| TF-IDF + Logistic Regression |     98.51% |     98.90% |      97.79% |     98.34% |
| **DistilBERT**               | **99.94%** | **99.86%** | **100.00%** | **99.93%** |

On this validation split, DistilBERT achieved higher accuracy, precision, recall, and F1-score than the TF-IDF + Logistic Regression baseline.

---

## 🔲 DistilBERT Confusion Matrix

```text
                 Predicted
                 REAL   FAKE

Actual REAL      4234      5
Actual FAKE         0   3491
```

### Interpretation

* **4,234** REAL articles were correctly classified.
* **5** REAL articles were classified as FAKE.
* **0** FAKE articles were classified as REAL.
* **3,491** FAKE articles were correctly classified.

---

## 📊 Model Comparison

![Model Comparison](results/model_comparison.png)

---

## 🔲 Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

---

# 🔍 Near-Duplicate Analysis

The validation accuracy was unusually high, so an additional TF-IDF cosine-similarity audit was performed.

The audit found:

| Similarity Threshold | Validation Articles |
| -------------------- | ------------------: |
| ≥ 0.90               |                 184 |
| ≥ 0.95               |                 153 |
| ≥ 0.99               |                 111 |

Most of the highly similar article pairs shared the same class label.

This indicates that near-duplicate overlap exists between the training and validation sets and should be considered when interpreting the validation results.

A separate similarity-aware retraining experiment was planned but could not be completed because GPU access became unavailable during the experiment.

Therefore, the reported **99.94% accuracy should be interpreted specifically as validation performance on this experimental split**, rather than expected real-world accuracy.

---

# 🖥️ Gradio Demo

The trained DistilBERT model was saved and later loaded for CPU-based inference.

A Gradio interface was created that allows users to enter a news article and receive:

* Predicted class
* REAL confidence
* FAKE confidence

![Gradio Demo](results/gradio_demo.png)

---

# 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **Hugging Face Transformers**
* **DistilBERT**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Gradio**
* **Google Colab**

---

# 📁 Project Structure

```text
fake-news-detection-distilbert/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── fake_news_detection.ipynb
│
├── src/
│   └── app.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── gradio_demo.png
│
├── docs/
│   └── project_report.pdf
│
└── models/
    └── README.md
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection-distilbert.git
```

Navigate into the project:

```bash
cd fake-news-detection-distilbert
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

The complete training and evaluation workflow is available in:

```text
notebooks/fake_news_detection.ipynb
```

The Gradio application can be launched using:

```bash
python src/app.py
```

> The trained model weights are not included directly in this repository because of their large file size. See the `models/` directory for model access instructions.

---

# ⚠️ Limitations

* The model learns patterns from the provided dataset rather than independently verifying claims.
* The dataset may contain source-specific or topic-specific biases.
* Near-duplicate articles were found across the train/validation split.
* Validation performance may therefore be higher than performance on completely unseen real-world news.
* The model may not generalize equally well to different news sources, topics, languages, or writing styles.
* Softmax confidence should not be interpreted as the probability that a real-world claim is objectively true or false.

---

# 🔮 Future Improvements

* Perform similarity-aware dataset splitting and retraining.
* Evaluate the model on an independent external dataset.
* Use a larger and more diverse news corpus.
* Investigate probability calibration.
* Add model explainability techniques.
* Deploy the model using Hugging Face Spaces.
* Add evidence-based external fact verification.
* Compare against additional Transformer architectures.

---

# 📄 Documentation

The project includes detailed documentation covering:

* Dataset preparation
* Tokenization
* DistilBERT architecture
* Training process
* Evaluation
* TF-IDF baseline
* Logistic Regression
* Near-duplicate analysis
* Model inference
* Gradio deployment

The complete project report is available in:

```text
docs/project_report.pdf
```

---

# 👨‍💻 Author

**Aryan**

B.Tech Computer Science Engineering

---

## ⭐ Project Highlights

* 38,646 unique news articles
* Transformer-based NLP classification
* DistilBERT fine-tuning
* TF-IDF + Logistic Regression baseline
* 99.94% validation accuracy
* Confusion matrix and classification metrics
* Near-duplicate similarity audit
* Saved model and CPU inference
* Interactive Gradio web interface

````


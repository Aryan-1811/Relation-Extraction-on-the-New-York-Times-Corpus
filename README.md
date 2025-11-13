# Relation-Extraction-on-the-New-York-Times-Corpus

This project builds and compares two models for Relation Extraction (RE) using the New York Times corpus:
1. **SVM with TF-IDF features**
2. **RoBERTa transformer model**

Both models can be run directly for inference using the saved model files.

## Overview
- SVM model provides a fast and interpretable baseline.
- RoBERTa achieves higher accuracy using contextual embeddings.
- The New York Times dataset is only required for training.
- Inference scripts allow you to input any sentence and receive a predicted relation instantly.

## Repository Structure
```
relation-extraction-nyt/
├── SVM_TextMining.ipynb
├── Roberta_TextMining.ipynb
├── inference_svm.py
├── inference_roberta.py
├── models/
│   ├── svm_text_classifier.pth
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   ├── relationship_extraction_model.pth
│   └── tokenizer.pkl
├── requirements.txt
└── README.md
```
## Installation
pip install -r requirements.txt

## Model Setup
Create a folder named `models` and place the following files inside it.

### SVM files
- `svm_text_classifier.pth`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`

### RoBERTa files
- `relationship_extraction_model.pth`
- `tokenizer.pkl`
- `label_encoder.pkl`

RoBERTa model weights download link:  
https://drive.google.com/file/d/1Fn9BPoTrG7nn1GPISTiZt-Lfz3xBHp21/view?usp=sharing

## Running Inference

### SVM Inference
python inference_svm.py
Enter any sentence when prompted and the predicted relation label will be displayed.

### RoBERTa Inference
python inference_roberta.py
Enter any sentence when prompted to receive the predicted relation.

## Reproducing Training (Optional)
- Open `SVM_TextMining.ipynb` to retrain the TF-IDF + SVM baseline.
- Open `Roberta_TextMining.ipynb` to fine-tune RoBERTa.

Both notebooks load NYT data, preprocess text, encode labels, train models, and evaluate performance.

## Notes
- The New York Times dataset is not included in this repository.
- Inference does not require the dataset.
- If a GPU is unavailable, both inference scripts automatically run on CPU.

# VisionIQ: An Advanced Visual Question Answering System

VisionIQ is a deep learning-based Visual Question Answering (VQA) system that integrates **CNNs**, **LSTMs**, and **Transformer-based multimodal fusion** to accurately answer text-based questions related to images. The system is designed to extract visual and textual features and combine them intelligently to understand and respond to visual questions effectively.

## 🔍 Overview

VisionIQ tackles the problem of answering natural language questions based on image content—particularly focusing on **text within images**. The system is evaluated on the **TextVQA dataset** and achieves a **95.4% accuracy**, outperforming traditional CNN-LSTM baselines.

## 🧠 Key Features

- **Convolutional Neural Networks (CNNs)** for visual feature extraction (ResNet-50)
- **LSTM** for sequential question encoding using GloVe embeddings
- **Transformer-based multimodal fusion** for joint understanding of image and question
- **OCR-based alignment** for text extraction and spatial correlation
- Attention mechanisms for interpretability and focus on relevant image/question parts

## 🛠️ Architecture

The system is composed of three main modules:

1. **Visual Feature Extraction**: ResNet-50 pretrained and fine-tuned on TextVQA
2. **Textual Encoding**: GloVe embeddings processed by LSTM
3. **Multimodal Fusion**: Transformer encoder with multi-head self-attention

## 🧪 Evaluation

| Model                 | Accuracy (%) |
|----------------------|--------------|
| CNN-LSTM Baseline    | 78.9         |
| **VisionIQ (Full)**  | **95.4**     |

## 📦 Technologies Used

- Python
- PyTorch / TensorFlow (specify your choice)
- ResNet-50
- LSTM
- Transformer (Self-Attention)
- GloVe Embeddings
- OCR (Optical Character Recognition)
- TextVQA Dataset

## 📁 Dataset

The model is evaluated using the **TextVQA** dataset, which includes images with embedded text (e.g., signs, labels) and natural language questions about the textual content in the image.

## 🚀 Future Work

- Multilingual VQA support using cross-lingual transformers
- Lightweight model versions for edge and mobile deployment
- Scene graph-based semantic relationship parsing


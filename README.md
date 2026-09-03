## 🤟 Real-Time American Sign Language Interpreter using Transformer and LLM
A deep learning-based, real-time system to interpret isolated American Sign Language (ASL) signs and generate fluent English sentences using a hybrid architecture combining Transformer Encoders and GPT-4.

---

## 📌 Project Overview
This project addresses the gap in real-time ASL interpretation by developing a two-stage pipeline:

Sign-to-Gloss Recognition: Recognizes isolated signs from video using MediaPipe and a Transformer Encoder.

Gloss-to-Text Generation: Converts predicted glosses into semantically meaningful sentences using GPT-4.

The system is lightweight, scalable, and deployable on standard CPUs—making it suitable for real-time inference without specialized hardware.

---

## 🧠 Key Features
⚡ Real-time inference using webcam input

🎯 Transformer Encoder for sign classification (250 isolated ASL signs)

🧾 GPT-4 prompting for fluent sentence generation

🛠️ MediaPipe Holistic model for landmark extraction (hand, face, pose)

📊 Evaluation using WER (Word Error Rate) and BLEU score

🖥️ Compatible with CPU hardware for deployment

---

## 📂 Dataset
Name: Google Isolated Sign Language Recognition (GISLR)

Size: 94,477 samples from 21 participants

---

## 🔍 Model Architecture
Transformer Encoder (Sign-to-Gloss)
Input: [135, num_landmarks * 2] shaped tensor

Positional encoding + CLS token

2-layer MLP for embedding

4 Encoder blocks, 11 heads, dropout = 0.318

Output: 250-class softmax (sign class prediction)

GPT-4 Prompting (Gloss-to-Text)
Input: Predicted glosses from Transformer

Custom prompt structure for visual, ASL-style sentence construction

Output: Fluent, 5-word ASL-aligned English sentence

---

## 📈 Evaluation Metrics

| Stage         | Metric | Score     |
| ------------- | ------ | --------- |
| Sign-to-Gloss | WER    | 72.85%    |
| Gloss-to-Text | BLEU-1 | **70.71** |
| Gloss-to-Text | BLEU-4 | **13.41** |

Compared to:

Camgoz et al. (BLEU-1: 47.26, BLEU-4: 22.38)

Sincan et al. (BLEU-4: 9.12)

---
## 🧪 Training Details
Framework: PyTorch

Optimizer: AdamW, LR = 0.0005

Scheduler: CosineAnnealingWarmRestarts

Regularization: Gradient Clipping (0.5), Dropout (0.318)

Epochs: 300–500

Tuning: Optuna with 20 trials

GPU: Trained on RTX 4090 (≈17 hours)

## 📚 References
MediaPipe: https://google.github.io/mediapipe/

GISLR Dataset: https://www.kaggle.com/competitions/asl-signs

Camgoz et al., CVPR 2020

Sincan et al., arXiv 2024

GPT-4: OpenAI API


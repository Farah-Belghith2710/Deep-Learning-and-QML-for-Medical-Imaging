# Hybrid Deep Learning + Quantum ML for Medical Imaging

> A lightweight, interactive web application combining a custom NumPy CNN feature extractor with a PennyLane variational quantum circuit classifier to detect synthetic medical scan anomalies.



---

## Disclaimer

**This is an educational demo, not a medical device.** All scans are computer-generated synthetic patterns used solely to demonstrate hybrid classical-quantum machine learning workflows. No real medical data is used or analyzed.

---

## Project Overview

This project demonstrates a two-stage **Hybrid Classical-Quantum Machine Learning** architecture:

1. **Classical Feature Extraction (CNN Backbone)**: A 2D Convolutional Neural Network built completely from scratch in pure NumPy (with manual backpropagation). It reduces input scans down to a **4-dimensional bottleneck vector**.
2. **Quantum Classification (PennyLane QML Head)**: A 4-qubit variational quantum circuit that takes the 4-dim CNN embeddings, angle-encodes them into quantum states, and optimizes entangling gates to classify scans as **Normal** vs. **Tumor**.

---

## Key Features

- **Custom NumPy CNN Backbone**:
  - `Conv2D(8 filters, 3x3)` → `ReLU` → `MaxPool2x2` → `Dense(16)` → `ReLU` → `Dense(4) + Tanh` (Bottleneck) → `Dense(1) + Sigmoid`.
- **PennyLane Quantum Classifier**:
  - 4 qubits with `AngleEmbedding` and `BasicEntanglerLayers`.
  - Trained via `NesterovMomentumOptimizer` tracking PauliZ expectation values on Qubit 0.
- **Interactive Streamlit Dashboard**:
  - **Dataset Preview**: Visualizer for synthetic grayscale tissue scans.
  - **Training & Results**: Real-time tracking for CNN loss/accuracy, 2D latent space embedding plots, quantum circuit diagram, and confusion matrix.
  - **Try It Yourself Playground**: Interactive parameter controls (mass location, radius, brightness, background noise) to generate custom scans and test live predictions with confidence scores.

---

## Repository Structure

```text
.
├── medical_qml_model.py  # Core QML model, NumPy CNN, dataset generator, training logic
├── app.py                # Streamlit UI dashboard and visualization tabs
├── requirements.txt      # Python dependencies
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation

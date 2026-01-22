<div align="center">

# 🔢 DigitVision: MNIST Neural Classifier
### *Deep Learning Framework for High-Precision Handwritten Digit Recognition*

---

[![Overview](https://img.shields.io/badge/📖_Overview-blue?style=for-the-badge)](#-project-overview)
[![Key Features](https://img.shields.io/badge/✨_Key_Features-6f42c1?style=for-the-badge)](#-key-features)
[![Tech Stack](https://img.shields.io/badge/🛠️_Tech_Stack-success?style=for-the-badge)](#-tech-stack)
[![Architecture](https://img.shields.io/badge/🏗️_Architecture-orange?style=for-the-badge)](#-technical-architecture)
[![Installation](https://img.shields.io/badge/🚀_Installation-red?style=for-the-badge)](#-installation--getting-started)
[![Contact](https://img.shields.io/badge/📩_Contact-lightgrey?style=for-the-badge)](#-contact)

---

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Deep Learning](https://img.shields.io/badge/AI-Deep--Learning-005850?style=flat-square)](https://en.wikipedia.org/wiki/Deep_learning)
[![Codiom](https://img.shields.io/badge/Powered_By-Codiom-FF4B4B?style=flat-square)](https://codiom.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-4caf50?style=flat-square)](https://opensource.org/licenses/MIT)

**Leveraging Convolutional Neural Networks to decode handwritten patterns with neural precision.**

</div>

---

## 📖 Project Overview

The **DigitVision System** is a sophisticated Deep Learning solution designed to classify handwritten digits from the iconic MNIST dataset. Developed as a foundational Computer Vision project within the **Codiom** initiative, this repository implements a high-performance neural architecture using **TensorFlow** and **Keras**.

As a Software Engineering student at Istanbul Aydın University, I designed this project to master the complexities of neural layer stacking, activation functions, and backpropagation in real-world AI applications.

---

## ✨ Key Features

* **🧠 Advanced Neural Architecture:** Implementation of a multi-layer Neural Network (MLP/CNN) optimized for image classification.
* **🛠️ Image Preprocessing Pipeline:**
    * **Normalization:** Scaling pixel values to the $[0, 1]$ range for faster gradient descent.
    * **Reshaping:** Automated tensor transformation to fit neural input layers.
    * **Data Augmentation:** Potential for expanding datasets via rotation and shifting.
* **🤖 Optimization Engine:** Utilizing **Adam** or **SGD** optimizers with categorical cross-entropy loss for maximum accuracy.
* **💾 Model Persistence:** Exporting trained weights into `.h5` or `SavedModel` formats for immediate production use.
* **📊 Visual Feedback:** Real-time generation of training/validation loss curves and accuracy metrics.

---

## 🛠️ Tech Stack

| Category | Technology | Usage |
| :--- | :--- | :--- |
| **Development** | **Python 3.9+** | Core logic and model orchestration. |
| **AI Framework** | **TensorFlow / Keras** | Primary engine for building and training neural layers. |
| **Data Processing**| **NumPy** | High-performance tensor and matrix operations. |
| **Visual Analytics**| **Matplotlib** | Visualizing digit samples and performance graphs. |
| **Version Control** | **Git** | Managing research revisions and codebase. |

---

## 🏗️ Technical Architecture

The system follows a standard **Deep Learning Workflow**, ensuring deterministic results and efficient resource utilization.



### Mathematical Validation

Model performance is quantified using cross-entropy loss and accuracy metrics:

* **Categorical Cross-Entropy:** Measures the performance of a classification model whose output is a probability value between 0 and 1.
* **Accuracy:**
  $$Accuracy = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

---

## 📂 Project Structure

```bash
.
├── 📁 data/                 # Raw MNIST dataset storage
├── 📄 train_model.py        # Core Deep Learning pipeline
├── 📄 predict.py            # Inference script for new digit samples
├── 📁 models/               # Saved neural weights (.h5)
├── 📄 requirements.txt      # Dependency manifest
└── 📄 README.md             # Documentation
```

## 🚀 Installation & Getting Started

### 1. Environment Preparation

```bash
# Clone the architecture
git clone [https://github.com/BerattCelikk/Tensarflow_Mnist_Predict_Digit.git](https://github.com/BerattCelikk/Tensarflow_Mnist_Predict_Digit.git)
cd Tensarflow_Mnist_Predict_Digit

# Initialize virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

```

### 2. Dependency Injection

```bash
pip install -r requirements.txt
```

### 3. Execution Flow
To train the neural network:
```bash
python train_model.py
```
To predict a digit from an image:
```bash
python predict.py --image sample_digit.png
```

## 🗺️ Roadmap

- [ ] CNN Integration: Migrating to Convolutional Neural Networks for state-of-the-art accuracy.
- [ ] Web Integration: Developing a canvas-based web interface using Flask to draw and predict digits in real-time.
- [ ] Mobile Deployment: Converting the model to TensorFlow Lite for mobile AI applications.
- [ ] Custom OCR: Expanding the model to recognize full handwritten sentences.

---

<div align="center" id="contact">

Architected with precision by Berat Erol Çelik Founder of Codiom

Software Engineering @ Istanbul Aydın University

</div>



















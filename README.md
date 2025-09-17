
# Custom VGG16 Model for Handwritten Digit Recognition (OCR)

## 📌 Objective

This project aims to advance handwritten digit recognition using a **Custom VGG16-based Convolutional Neural Network (CNN)** architecture. The model was trained and optimized on the **MNIST dataset**, a widely recognized benchmark for OCR (Optical Character Recognition), achieving **99.36% test accuracy**.

---

## 🚀 Features

* Custom **VGG16-inspired CNN** architecture
* Layers: **ReLU**, **Max Pooling**, **Batch Normalization**, **Dropout**
* Robust **training pipeline** with **data augmentation**
* Implemented fully in **PyTorch**
* High performance on MNIST with **state-of-the-art accuracy**

---

## 🛠️ Tech Stack

* **Language:** Python
* **Framework:** PyTorch
* **Dataset:** MNIST (handwritten digit dataset)
* **Libraries:** NumPy, Matplotlib, TorchVision

---

## 📂 Project Structure

```
├── data/                  # MNIST dataset (downloaded via torchvision)
├── models/                # Custom VGG16 model definition
│   └── custom_vgg16.py
├── notebooks/             # Jupyter notebooks for experimentation
├── utils/                 # Helper functions (training, evaluation, visualization)
├── train.py               # Script to train the model
├── evaluate.py            # Script to evaluate model accuracy
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## 📊 Results

* **Test Accuracy:** 99.36%
* Model trained with **data augmentation** (rotation, scaling, and shifting) for better generalization.
* Achieved robust digit classification with minimal misclassifications.

Sample prediction:

| Input Image                                                                     | Predicted Label       |
| ------------------------------------------------------------------------------- | --------------------- |
| ![digit](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png) | ✅ Correct (e.g., `5`) |

---

## 🔧 Installation & Usage

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/custom-vgg16-mnist.git
cd custom-vgg16-mnist
```

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Model

```bash
python train.py --epochs 20 --batch_size 64
```

### 4️⃣ Evaluate Model

```bash
python evaluate.py --model_path checkpoints/vgg16_mnist.pth
```

---

## 📈 Future Improvements

* Extend to **EMNIST dataset** (letters + digits)
* Apply **transfer learning** on real-world handwritten text datasets
* Deploy as an **OCR microservice API**

---

## 🙌 Acknowledgements

* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
* [PyTorch](https://pytorch.org/)

---


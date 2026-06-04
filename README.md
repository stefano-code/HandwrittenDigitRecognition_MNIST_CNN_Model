# Handwritten Digit Recognition with CNN and TensorFlow Lite

A complete example of handwritten digit recognition using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset** with **TensorFlow/Keras**.

This project demonstrates the entire machine learning workflow:

* Loading and exploring the MNIST dataset
* Building a Convolutional Neural Network
* Training and evaluating the model
* Converting the trained model to TensorFlow Lite
* Generating a quantized TensorFlow Lite model for mobile and embedded devices
* Preparing the model for Android deployment

---

## Project Overview

Handwritten digit recognition is one of the most popular introductory computer vision problems.

<img width="2842" height="1127" alt="mnist" src="https://github.com/user-attachments/assets/76de3d47-bb06-49eb-ba32-90737b8bb533" />

In this project, a CNN is trained to classify grayscale images of handwritten digits into one of ten classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

The model is trained on the MNIST dataset and achieves approximately **99% test accuracy**.

Two TensorFlow Lite models are generated:

* Standard TensorFlow Lite model
* Quantized TensorFlow Lite model

Both models can be deployed directly on Android devices using the TensorFlow Lite runtime.

---

## Dataset

The project uses the MNIST dataset provided by Keras.

### Training Set

* 60,000 grayscale images
* Image size: 28 × 28 pixels
* 10 digit classes

### Test Set

* 10,000 grayscale images
* Image size: 28 × 28 pixels
* 10 digit classes

---

## CNN Architecture

The network consists of three convolutional layers followed by a fully connected classifier.

```text
Input (28x28x1)

├── Conv2D (32 filters, 3x3, ReLU)
├── MaxPooling2D (2x2)

├── Conv2D (64 filters, 3x3, ReLU)
├── MaxPooling2D (2x2)

├── Conv2D (64 filters, 3x3, ReLU)

├── Flatten

├── Dense (64 units, ReLU)

└── Dense (10 units, Softmax)
```

### Model Statistics

| Parameter            | Value       |
| -------------------- | ----------- |
| Total Parameters     | 93,322      |
| Trainable Parameters | 93,322      |
| Input Shape          | 28 × 28 × 1 |
| Output Classes       | 10          |

---

## Data Preprocessing

### Image Processing

The original images have shape:

```python
(28, 28)
```

Before training, images are reshaped to include the channel dimension:

```python
(28, 28, 1)
```

Pixel values are then normalized:

```python
image = image.astype("float32") / 255.0
```

This scales all pixel values from:

```text
[0, 255] → [0, 1]
```

which helps the network train more efficiently.

---

### Label Processing

The labels are converted into one-hot encoded vectors:

```python
from keras.utils import to_categorical
```

Example:

```text
Digit 3

[0,0,0,1,0,0,0,0,0,0]
```

This representation matches the output of the Softmax layer.

---

## Model Compilation

The network is compiled using:

```python
model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Configuration

| Setting       | Value                    |
| ------------- | ------------------------ |
| Optimizer     | RMSprop                  |
| Loss Function | Categorical Crossentropy |
| Metric        | Accuracy                 |

---

## Training

The model is trained using:

```python
model.fit(
    train_images_cnn,
    train_labels_cnn,
    epochs=5,
    batch_size=60
)
```

### Training Parameters

| Parameter        | Value  |
| ---------------- | ------ |
| Epochs           | 5      |
| Batch Size       | 60     |
| Training Samples | 60,000 |

---

## Results

After training, the model achieves approximately:

| Metric            | Value  |
| ----------------- | ------ |
| Training Accuracy | ~99.4% |
| Test Accuracy     | ~99.0% |

These results demonstrate the effectiveness of CNNs for image classification tasks.

---

## TensorFlow Lite Conversion

The trained Keras model is converted to TensorFlow Lite:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model_cnn)
tflite_model = converter.convert()
```

Output file:

```text
digit_recognition_cnn.tflite
```

Approximate size:

```text
377 KB
```

---

## TensorFlow Lite Quantization

To reduce model size and improve deployment efficiency, a quantized version is also generated:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

Output file:

```text
digit_recognition_cnn_quant.tflite
```

Approximate size:

```text
104 KB
```

### Benefits of Quantization

* Smaller model size
* Reduced memory usage
* Faster inference on many embedded devices
* Easier deployment on resource-constrained hardware

---

## Android Deployment

Both generated TensorFlow Lite models can be integrated directly into Android applications.

### Supported Models

```text
digit_recognition_cnn.tflite
digit_recognition_cnn_quant.tflite
```

### Expected Input

```text
Shape: 28 x 28 x 1
Type: float32
Range: [0,1]
```

### Output

The model returns a probability distribution across the ten digit classes:

```text
[0 ... 9]
```

The predicted digit corresponds to the class with the highest probability.

---

## Requirements

Install the required Python packages:

```bash
pip install tensorflow keras
```

---

## Running the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

Then execute all cells sequentially.

The notebook will:

1. Download the MNIST dataset
2. Build the CNN architecture
3. Train the model
4. Evaluate model accuracy
5. Generate a TensorFlow Lite model
6. Generate a quantized TensorFlow Lite model

---

## Project Structure

```text
project/
│
├── digit_recognition.ipynb
├── digit_recognition_cnn.tflite
├── digit_recognition_cnn_quant.tflite
├── README.md
└── requirements.txt
```

---

## Learning Objectives

This project is intended as an educational example covering:

* Convolutional Neural Networks (CNNs)
* Image preprocessing
* Multi-class classification
* TensorFlow/Keras workflows
* TensorFlow Lite conversion
* Model quantization
* Mobile AI deployment

---

## License

This project is provided for educational and research purposes.

Feel free to modify, extend, and use the code in your own projects.


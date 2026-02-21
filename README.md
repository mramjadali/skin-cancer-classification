# Skin Cancer Classification using CNN

This project is a **binary skin cancer classifier** using Convolutional Neural Networks (CNN) implemented in TensorFlow/Keras. The model predicts whether a skin lesion image is **benign** or **malignant**.

---

## Dataset

The dataset used is **[Skin Cancer Malignant vs Benign](https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign)** downloaded from Kaggle.  

- **Total images:** ~1,000+  
- **Classes:** `benign`, `malignant`  
- **Structure:**  
<img width="426" height="325" alt="image" src="https://github.com/user-attachments/assets/e712a1e8-12e6-41da-8fe5-ca12f5bcfb2a" />


---

## Project Structure
<img width="1011" height="278" alt="image" src="https://github.com/user-attachments/assets/48eb0784-ebcc-4f39-85ba-678c063e88fe" />


---

## Features

- ✅ CNN model with 3 convolutional layers + max pooling + dense layers  
- ✅ Binary classification: Benign vs Malignant  
- ✅ Image preprocessing and augmentation (rotation, zoom, horizontal flip)  
- ✅ Upload new images to test predictions easily
  <img width="1161" height="735" alt="image" src="https://github.com/user-attachments/assets/767eb02f-8d87-4e5d-a824-7fc681e6c4c4" />


---

## Usage

1. Open `skin_cancer_notebook.ipynb` in Google Colab  
2. Run all cells to train or load the model  
3. Upload a new skin lesion image to predict its class:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load image
img = image.load_img("your_image.jpg", target_size=(128,128))
img_arr = image.img_to_array(img)/255.0
img_arr = np.expand_dims(img_arr, axis=0)

# Predict
pred = model.predict(img_arr)
if pred[0][0] > 0.5:
    print("Prediction: Malignant")
else:
    print("Prediction: Benign")


## 👨‍💻 Role
I implemented the entire project, including dataset preprocessing, building the CNN, training the model, and testing predictions on uploaded images. I also prepared the notebook and structured the project for easy reproducibility.
**📊 Results**
Example prediction:
Uploaded image: 1003.jpg
Prediction: Benign
Training Accuracy: ~95–97% (depending on epochs and augmentation)

Confusion matrix and classification reports available in the notebook


## 📦 Requirements
  Python 3.x

  TensorFlow / Keras

  NumPy

  Matplotlib

  scikit-learn
## 📜 License
This project is for educational purposes and university submission.
## 🔗 Links**
GitHub Repository: https://github.com/mramjadali/skin-cancer-classification
Dataset Source: Kaggle - Skin Cancer Malignant vs Benign



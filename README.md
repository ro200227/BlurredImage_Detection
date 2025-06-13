# Blurred Image Detection using ML

This project is focused on classifying images into three categories:  
**Sharp**  
**Defocused Blurred**  
**Motion Blurred**  

We use handcrafted features like Laplacian variance, FFT frequency components, edge density, and Sobel gradient statistics, followed by a machine learning model (Random Forest) for classification.

---

## Dataset

The dataset used is from [Kaggle - Blur Dataset by Kwentar](https://www.kaggle.com/datasets/kwentar/blur-dataset), containing:

- `sharp/` (350 images)  
- `defocused_blurred/` (350 images)  
- `motion_blurred/` (350 images)  

Each image is resized to `256x256` before feature extraction.

---

## Features Used

- **Laplacian Variance** – Detects overall sharpness.
- **FFT Mean of Central Frequencies** – Captures frequency domain characteristics.
- **Edge Density (Canny)** – Measures presence of edges.

---

## Model

- **Algorithm**: `RandomForestClassifier`
- **Evaluation**:
  - Accuracy: **~89%**
  - Good performance especially for `sharp` and `defocused`
- **Confusion Matrix** & classification report included

---

## Tech Stack

 -Python, OpenCV, NumPy
 
 -Scikit-learn (RandomForestClassifier)
 
 -Streamlit (for UI)
 
 -Matplotlib


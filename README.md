📌 Emotion Recognition with Compact Convolutional Transformers (CCT)

Real-time facial emotion detection using a Custom Compact Convolutional Transformer (CCT) trained on the FER2013 dataset.

This project performs:

📷 Real-time webcam emotion recognition

🧠 Transformer-based image classification (CCT)

🔍 Seven FER2013 emotion classes

📊 Performance evaluation (confusion matrix & metrics)

🚀 Features

Compact Convolutional Transformer (CCT) model

Convolutional tokenizer

Transformer encoder layers

MLP projection + GELU activation

Preprocessing pipeline

Normalization

Facial extraction

CLAHE contrast enhancement

Optional sharpening filter

Real-time inference

OpenCV-based webcam streaming

Face detection using Haar cascades

Live emotion prediction overlay

🎭 Emotion Classes

The model predicts 7 standard FER2013 emotion labels:

Label	Emotion
0	Angry
1	Disgust
2	Fear
3	Happy
4	Sad
5	Surprise
6	Neutral
🧩 Model Architecture (CCT Overview)

CCT-optimized structure:

Input → Data Augmentation

Conv2D tokenizer

Transformer blocks

Multi-head self-attention

Layer normalization

MLP feed-forward network

Global average pooling

Softmax classifier

CCT models are efficient and well suited for small images like 48×48 FER2013.

📁 Repository Structure
📦 emotion-recognition-cct
 ┣ 📂 models/
 ┃ ┗ my_cct_model.keras     # Saved model (optional)
 ┣ 📂 notebooks/
 ┃ ┗ training.ipynb         # Model training + evaluation
 ┣ 📂 src/
 ┃ ┗ realtime_emotion.py    # Webcam real-time script
 ┣ README.md
 ┗ requirements.txt

🏋️ Training Details

Dataset: FER2013
Input Shape: (48, 48, 1)
Training method:

Image normalization

Optional augmentation: flip, rotation, zoom, brightness/contrast jitter

Class balancing using computed class weights

Adam optimizer

Categorical crossentropy loss

Performance:
The model achieves ~50% baseline accuracy (expected for FER2013).
With class-balancing and augmentation, performance can reach 58–65%.

🧪 Evaluation

Evaluation includes:

Confusion matrix

Precision / Recall / F1-score

Accuracy curves

Loss curves

Example metrics:

Accuracy: ~0.50 (baseline)
Best class: Happy
Hardest classes: Disgust, Fear

🎥 Real-Time Detection

Run:

python realtime_emotion.py


Features:

Auto face detection

Preprocessing to match FER2013

Live bounding box and emotion label

Quit with Q

🛠 Requirements

Add this to requirements.txt:

tensorflow
opencv-python
numpy
matplotlib
seaborn
scikit-learn

▶️ How to Run
1. Clone the repository
git clone https://github.com/yourusername/emotion-recognition-cct.git
cd emotion-recognition-cct

2. Install dependencies
pip install -r requirements.txt

3. Run training (optional)
jupyter notebook ./notebooks/training.ipynb

4. Run real-time emotion recognition
python src/realtime_emotion.py

📌 Notes

FER2013 is a challenging dataset and common models often overfit to "Sad" or "Neutral" without balancing or augmentation.

Real-time performance varies depending on lighting and camera quality.

Future improvements include:

Better augmentation strategies

Larger/more robust transformer models

Face alignment

EMA or label smoothing

🙌 Acknowledgements

FER2013 dataset (Kaggle)

Compact Convolutional Transformers (Hassani et al., 2021)

TensorFlow & OpenCV

📜 License

MIT License

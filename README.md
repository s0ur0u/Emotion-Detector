📌 Emotion Recognition with Compact Convolutional Transformers (CCT)

Real-time facial emotion detection using a Custom Compact Convolutional Transformer (CCT) trained on the FER2013 dataset.

This project performs:
     📷 Real-time webcam emotion recognition
     🧠 Transformer-based image classification (CCT)
     🔍 Seven FER2013 emotion classes
     📊 Performance evaluation (confusion matrix & metrics)
   

🚀 Features
Compact Convolutional Transformer (CCT) model
    -> Convolutional tokenizer
    -> Transformer encoder layers
    -> MLP projection + GELU activation

Preprocessing pipeline
    -> Normalization
    -> Facial extraction
    -> CLAHE contrast enhancement
    -> Optional sharpening filter

Real-time inference
    -> OpenCV-based webcam streaming
    -> Face detection using Haar cascades
    -> Live emotion prediction overlay


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
 -Input → Data Augmentation
 -Conv2D tokenizer
 -Transformer blocks
    ->Multi-head self-attention
    ->Layer normalization
    ->MLP feed-forward network
 -Global average pooling
 -Softmax classifier
CCT models are efficient and well-suited for small images like 48×48 FER2013.

📌 Notes
- FER2013 is a challenging dataset and common models often overfit to "Sad" or "Neutral" without balancing or augmentation.
- Real-time performance varies depending on lighting and camera quality.
- Future improvements include:
    -> Better augmentation strategies
    -> Larger/more robust transformer models
    -> Face alignment
    -> EMA or label smoothing

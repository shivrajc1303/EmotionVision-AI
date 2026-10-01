\# EmotionVision AI 🎭



Real-Time Facial Emotion Recognition System using CNN, TensorFlow/Keras and OpenCV.



\## 📌 Overview



EmotionVision AI is a computer vision and deep learning project that detects human facial emotions in real time through a webcam.



The system uses a custom Convolutional Neural Network (CNN) trained on the FER-2013 dataset to classify facial expressions into seven emotion categories.



\## 😊 Emotion Classes



\- Angry

\- Disgust

\- Fear

\- Happy

\- Neutral

\- Sad

\- Surprise



\## 🚀 Features



\- Real-time facial emotion recognition

\- Webcam-based face detection

\- CNN-based emotion classification

\- Seven emotion categories

\- Confidence score for predictions

\- Tkinter desktop GUI

\- Model evaluation using accuracy, precision, recall and F1-score

\- Confusion matrix visualization

\- Emotion statistics dashboard

\- Screenshot capture functionality



\## 🧠 Technologies Used



\- Python

\- TensorFlow / Keras

\- OpenCV

\- NumPy

\- Scikit-learn

\- Matplotlib

\- Seaborn

\- Tkinter

\- Pillow



\## 📊 Dataset



The model was trained using the FER-2013 facial expression dataset.



Images are converted to grayscale and resized to:



```text

48 × 48 pixels 



The input format used by the CNN is:



48 × 48 × 1



🏗️ CNN Architecture



The model consists of:



Convolutional layers

Batch Normalization

Max Pooling

Dropout

Fully Connected Dense Layer

Softmax output layer



The final layer contains 7 neurons corresponding to the seven emotion classes.



📈 Model Performance



The baseline CNN achieved approximately:



Test Accuracy: \~60%



Evaluation metrics include:



Accuracy

Precision

Recall

F1-score

Confusion Matrix



Detailed evaluation results are available in the results/ directory.



🖥️ Application



The project includes a Tkinter-based desktop application with:



Live webcam preview

Face detection

Emotion prediction

Confidence score

Emotion statistics

Capture functionality

Start/Stop camera controls


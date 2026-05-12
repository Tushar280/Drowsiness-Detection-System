# 😴 Drowsiness Detection System

A real-time computer vision application that monitors a user's face to detect signs of drowsiness, such as closed eyes, yawning, and head tilts, sounding an alarm to prevent accidents.

## ✨ Features

- **Real-Time Monitoring:** Captures live video feed using your webcam.
- **Facial Landmark Detection:** Utilizes MediaPipe Face Mesh for precise tracking of the eyes and mouth.
- **Deep Learning Classification:** Uses a custom TensorFlow/Keras model to classify "Open", "Closed", and "Yawning" states.
- **Head Tilt Detection:** Calculates tilt angles to detect posture loss.
- **Audio Alerts:** Triggers a warning beep when drowsiness thresholds are exceeded.

## 🖥️ Hardware Requirements

To run this project, you will need the following hardware:

1. **Computer/Laptop:** Windows OS is required for the default audio alerts (`winsound`).
2. **Webcam:** A built-in or external USB camera.
3. **Speakers:** Audio output device to hear the warning alarms.

## ⚙️ How to Run

Follow these steps to run the project on your local machine:

**1. Clone the repository**
```bash
git clone https://github.com/Tushar280/Drowsiness-Detection-System.git
cd Drowsiness-Detection-System
```

**2. Install dependencies**
Ensure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

**3. Start the system**
The pre-trained model is already included in the repository. Simply run the main script:
```bash
python app.py
```
> **Note:** To exit the camera window, press the **'q'** key.

## 🧠 Custom Training (Optional)

If you wish to train the model from scratch on your own data, run the training script:
```bash
python train.py
```

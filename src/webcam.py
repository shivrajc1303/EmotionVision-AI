import os
import sys

import cv2
import numpy as np

from tensorflow.keras.models import load_model  # type: ignore[reportMissingImports]


ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(ROOT_DIR)


MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "emotion_model.keras"
)


EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


print("\nLoading EmotionVision AI model...")

model = load_model(
    MODEL_PATH
)


print("Model loaded successfully!")


face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("\nError: Could not open webcam.")

    sys.exit()


print("\nWebcam started!")
print("Press ESC to exit.")


while True:

    success, frame = camera.read()

    if not success:
        print("Failed to read webcam.")
        break


    frame = cv2.flip(
        frame,
        1
    )


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )


    for x, y, w, h in faces:

        face = gray[
            y:y+h,
            x:x+w
        ]


        face = cv2.resize(
            face,
            (48, 48)
        )


        face = face.astype(
            "float32"
        ) / 255.0


        face = np.expand_dims(
            face,
            axis=0
        )


        face = np.expand_dims(
            face,
            axis=-1
        )


        prediction = model.predict(
            face,
            verbose=0
        )[0]


        emotion_index = np.argmax(
            prediction
        )


        emotion = EMOTIONS[
            emotion_index
        ]


        confidence = prediction[
            emotion_index
        ] * 100


        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        text = (
            f"{emotion}: "
            f"{confidence:.1f}%"
        )


        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    cv2.putText(
        frame,
        "EmotionVision AI",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Press ESC to exit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )


    cv2.imshow(
        "EmotionVision AI - Real-Time Emotion Detection",
        frame
    )


    key = cv2.waitKey(1)


    if key == 27:
        break


camera.release()

cv2.destroyAllWindows()

print("\nEmotionVision AI stopped.")
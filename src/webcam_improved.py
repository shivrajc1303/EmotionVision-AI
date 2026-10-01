import os
import sys
from collections import deque, Counter

import cv2
import numpy as np

try:
    import tensorflow.keras as keras  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError):
    try:
        import keras  # type: ignore[import-not-found]
    except ImportError:
        keras = None

if keras is None:
    raise ImportError(
        "TensorFlow or Keras is required to run EmotionVision AI."
    )

load_model = keras.models.load_model


ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(ROOT_DIR)


MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "emotion_model_improved.keras"
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


CONFIDENCE_THRESHOLD = 0.35
SMOOTHING_FRAMES = 8


print("\n========================================")
print(" EmotionVision AI - Improved Webcam")
print("========================================")


print("\nLoading model...")

model = load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


print("\nLoading face detector...")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():

    print("ERROR: Face detector could not be loaded.")

    sys.exit()


print("Face detector loaded successfully!")


print("\nOpening camera...")

# IMPORTANT:
# This is the same camera configuration
# that worked in camera_test.py.

camera = cv2.VideoCapture(
    0,
    cv2.CAP_ANY
)


if not camera.isOpened():

    print("\nERROR: Camera could not be opened.")

    print("\nTry closing:")
    print("- Windows Camera")
    print("- Zoom")
    print("- Microsoft Teams")
    print("- Google Meet")
    print("- OBS")
    print("- Any other application using the webcam")

    sys.exit()


print("Camera opened successfully!")


emotion_history = deque(
    maxlen=SMOOTHING_FRAMES
)

confidence_history = deque(
    maxlen=SMOOTHING_FRAMES
)


print("\nStarting EmotionVision AI...")
print("Press ESC to exit.")


while True:

    success, frame = camera.read()


    if not success:

        print("\nERROR: Could not read camera frame.")

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
        f"Faces: {len(faces)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Improved CNN",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "ESC = Exit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
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


        current_emotion = EMOTIONS[
            emotion_index
        ]


        current_confidence = float(
            prediction[
                emotion_index
            ]
        )


        if current_confidence >= CONFIDENCE_THRESHOLD:

            emotion_history.append(
                current_emotion
            )

            confidence_history.append(
                current_confidence
            )


        if len(emotion_history) > 0:

            emotion_counts = Counter(
                emotion_history
            )

            stable_emotion = (
                emotion_counts
                .most_common(1)[0][0]
            )

        else:

            stable_emotion = "Uncertain"


        stable_confidences = []


        for emotion, confidence in zip(
            emotion_history,
            confidence_history
        ):

            if emotion == stable_emotion:

                stable_confidences.append(
                    confidence
                )


        if len(stable_confidences) > 0:

            stable_confidence = (
                sum(stable_confidences)
                /
                len(stable_confidences)
            )

        else:

            stable_confidence = 0


        display_confidence = (
            stable_confidence * 100
        )


        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        text = (
            f"{stable_emotion}: "
            f"{display_confidence:.1f}%"
        )


        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )


    cv2.imshow(
        "EmotionVision AI - Improved",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == 27:

        break


camera.release()

cv2.destroyAllWindows()


print(
    "\nEmotionVision AI stopped successfully."
)
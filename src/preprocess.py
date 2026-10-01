import os
import cv2
import numpy as np


IMG_SIZE = 48

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


def load_dataset(data_path):

    images = []
    labels = []

    for label, emotion in enumerate(EMOTIONS):

        emotion_path = os.path.join(
            data_path,
            emotion
        )

        if not os.path.exists(emotion_path):
            print(
                f"Folder not found: {emotion_path}"
            )
            continue

        for image_name in os.listdir(
            emotion_path
        ):

            image_path = os.path.join(
                emotion_path,
                image_name
            )

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            image = cv2.resize(
                image,
                (IMG_SIZE, IMG_SIZE)
            )

            images.append(image)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    images = images.astype(
        "float32"
    ) / 255.0

    images = np.expand_dims(
        images,
        axis=-1
    )

    return images, labels
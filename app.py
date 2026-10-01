import os
import cv2
import numpy as np
import tkinter as tk
import importlib

from PIL import Image, ImageTk
from tkinter import messagebox

try:
    load_model = importlib.import_module("tensorflow.keras.models").load_model
except ImportError:
    try:
        load_model = importlib.import_module("keras.models").load_model
    except ImportError:
        load_model = None


ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "emotion_model.keras"
)

SCREENSHOT_DIR = os.path.join(
    ROOT_DIR,
    "screenshots"
)

os.makedirs(
    SCREENSHOT_DIR,
    exist_ok=True
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


emotion_counts = {
    emotion: 0
    for emotion in EMOTIONS
}


model = load_model(
    MODEL_PATH
)


face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


camera = None
camera_running = False
current_frame = None
capture_number = 0


root = tk.Tk()

root.title(
    "EmotionVision AI"
)

root.geometry(
    "1200x760"
)

root.configure(
    bg="#111111"
)

root.resizable(
    False,
    False
)


title = tk.Label(
    root,
    text="EmotionVision AI",
    font=("Segoe UI", 28, "bold"),
    bg="#111111",
    fg="white"
)

title.pack(
    pady=(15, 2)
)


subtitle = tk.Label(
    root,
    text="Real-Time Facial Emotion Recognition System",
    font=("Segoe UI", 11),
    bg="#111111",
    fg="#999999"
)

subtitle.pack(
    pady=(0, 10)
)


main_frame = tk.Frame(
    root,
    bg="#111111"
)

main_frame.pack()


camera_frame = tk.Frame(
    main_frame,
    bg="#202020",
    width=700,
    height=500
)

camera_frame.grid(
    row=0,
    column=0,
    padx=15,
    pady=10
)

camera_frame.grid_propagate(
    False
)


video_label = tk.Label(
    camera_frame,
    text="Camera is stopped",
    font=("Segoe UI", 18),
    bg="#202020",
    fg="#777777"
)

video_label.pack(
    expand=True
)


dashboard = tk.Frame(
    main_frame,
    bg="#181818",
    width=400,
    height=500
)

dashboard.grid(
    row=0,
    column=1,
    padx=15,
    pady=10
)

dashboard.grid_propagate(
    False
)


detected_title = tk.Label(
    dashboard,
    text="CURRENT EMOTION",
    font=("Segoe UI", 11, "bold"),
    bg="#181818",
    fg="#888888"
)

detected_title.pack(
    pady=(25, 3)
)


emotion_label = tk.Label(
    dashboard,
    text="---",
    font=("Segoe UI", 32, "bold"),
    bg="#181818",
    fg="white"
)

emotion_label.pack(
    pady=3
)


confidence_label = tk.Label(
    dashboard,
    text="Confidence: 0.0%",
    font=("Segoe UI", 13),
    bg="#181818",
    fg="#bbbbbb"
)

confidence_label.pack(
    pady=5
)


face_label = tk.Label(
    dashboard,
    text="Faces detected: 0",
    font=("Segoe UI", 12),
    bg="#181818",
    fg="#bbbbbb"
)

face_label.pack(
    pady=5
)


separator = tk.Frame(
    dashboard,
    bg="#333333",
    height=1
)

separator.pack(
    fill="x",
    padx=25,
    pady=15
)


stats_title = tk.Label(
    dashboard,
    text="EMOTION STATISTICS",
    font=("Segoe UI", 11, "bold"),
    bg="#181818",
    fg="#888888"
)

stats_title.pack(
    pady=(0, 8)
)


stats_labels = {}

for emotion in EMOTIONS:

    label = tk.Label(
        dashboard,
        text=f"{emotion}: 0",
        font=("Segoe UI", 10),
        bg="#181818",
        fg="#cccccc",
        anchor="w"
    )

    label.pack(
        fill="x",
        padx=45,
        pady=2
    )

    stats_labels[emotion] = label


status_label = tk.Label(
    dashboard,
    text="Status: Camera Off",
    font=("Segoe UI", 10),
    bg="#181818",
    fg="#888888"
)

status_label.pack(
    pady=15
)


button_frame = tk.Frame(
    root,
    bg="#111111"
)

button_frame.pack(
    pady=10
)


start_button = tk.Button(
    button_frame,
    text="▶ Start Camera",
    font=("Segoe UI", 11, "bold"),
    width=17,
    command=lambda: start_camera()
)

start_button.grid(
    row=0,
    column=0,
    padx=5
)


stop_button = tk.Button(
    button_frame,
    text="■ Stop Camera",
    font=("Segoe UI", 11, "bold"),
    width=17,
    command=lambda: stop_camera()
)

stop_button.grid(
    row=0,
    column=1,
    padx=5
)


capture_button = tk.Button(
    button_frame,
    text="📸 Capture",
    font=("Segoe UI", 11, "bold"),
    width=17,
    command=lambda: capture_image()
)

capture_button.grid(
    row=0,
    column=2,
    padx=5
)


reset_button = tk.Button(
    button_frame,
    text="↻ Reset Stats",
    font=("Segoe UI", 11, "bold"),
    width=17,
    command=lambda: reset_statistics()
)

reset_button.grid(
    row=0,
    column=3,
    padx=5
)


def start_camera():

    global camera
    global camera_running

    if camera_running:
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        messagebox.showerror(
            "Camera Error",
            "Could not open webcam."
        )

        return

    camera_running = True

    status_label.config(
        text="Status: Camera Active",
        fg="#00cc88"
    )

    update_camera()


def stop_camera():

    global camera
    global camera_running

    camera_running = False

    if camera is not None:

        camera.release()

        camera = None

    video_label.config(
        image="",
        text="Camera is stopped"
    )

    video_label.image = None

    emotion_label.config(
        text="---"
    )

    confidence_label.config(
        text="Confidence: 0.0%"
    )

    face_label.config(
        text="Faces detected: 0"
    )

    status_label.config(
        text="Status: Camera Off",
        fg="#888888"
    )


def update_camera():

    global current_frame

    if not camera_running:
        return

    success, frame = camera.read()

    if not success:

        stop_camera()

        return

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

    detected_emotion = "---"
    detected_confidence = 0.0

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

        detected_emotion = EMOTIONS[
            emotion_index
        ]

        detected_confidence = (
            prediction[
                emotion_index
            ] * 100
        )

        emotion_counts[
            detected_emotion
        ] += 1

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        text = (
            f"{detected_emotion}: "
            f"{detected_confidence:.1f}%"
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


    emotion_label.config(
        text=detected_emotion
    )

    confidence_label.config(
        text=f"Confidence: {detected_confidence:.1f}%"
    )

    face_label.config(
        text=f"Faces detected: {len(faces)}"
    )


    for emotion in EMOTIONS:

        stats_labels[
            emotion
        ].config(
            text=(
                f"{emotion}: "
                f"{emotion_counts[emotion]}"
            )
        )


    current_frame = frame.copy()


    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    image = Image.fromarray(
        frame_rgb
    )


    image = image.resize(
        (680, 480)
    )


    photo = ImageTk.PhotoImage(
        image=image
    )


    video_label.config(
        image=photo,
        text=""
    )

    video_label.image = photo


    root.after(
        20,
        update_camera
    )


def capture_image():

    global capture_number

    if current_frame is None:

        messagebox.showwarning(
            "Capture",
            "Start the camera first."
        )

        return

    capture_number += 1

    filename = os.path.join(
        SCREENSHOT_DIR,
        f"emotion_capture_{capture_number}.jpg"
    )

    cv2.imwrite(
        filename,
        current_frame
    )

    messagebox.showinfo(
        "Screenshot Saved",
        f"Saved to:\n{filename}"
    )


def reset_statistics():

    for emotion in EMOTIONS:

        emotion_counts[
            emotion
        ] = 0

        stats_labels[
            emotion
        ].config(
            text=f"{emotion}: 0"
        )


def close_application():

    stop_camera()

    root.destroy()


root.protocol(
    "WM_DELETE_WINDOW",
    close_application
)


root.mainloop()
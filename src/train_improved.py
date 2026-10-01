import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight


ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(ROOT_DIR)


from src.preprocess import (
    load_dataset,
    EMOTIONS
)


TRAIN_PATH = os.path.join(
    ROOT_DIR,
    "dataset",
    "train"
)

TEST_PATH = os.path.join(
    ROOT_DIR,
    "dataset",
    "test"
)

MODEL_DIR = os.path.join(
    ROOT_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    ROOT_DIR,
    "results"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


print("\nLoading training dataset...")

X_train, y_train = load_dataset(
    TRAIN_PATH
)


print("\nLoading test dataset...")

X_test, y_test = load_dataset(
    TEST_PATH
)


print(
    "\nTraining images:",
    len(X_train)
)

print(
    "Testing images:",
    len(X_test)
)


print("\nCalculating class weights...")


class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)


class_weights = {
    i: weight
    for i, weight in enumerate(
        class_weights_array
    )
}


for i, emotion in enumerate(EMOTIONS):

    print(
        f"{emotion}: "
        f"{class_weights[i]:.3f}"
    )


print("\nCreating data augmentation...")


data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.08
    ),

    tf.keras.layers.RandomZoom(
        0.10
    ),

    tf.keras.layers.RandomContrast(
        0.10
    )

])


print("\nCreating improved CNN...")


model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(48, 48, 1)
    ),

    data_augmentation,


    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Dropout(
        0.25
    ),


    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Dropout(
        0.30
    ),


    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Dropout(
        0.35
    ),


    tf.keras.layers.Flatten(),


    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(
        0.5
    ),


    tf.keras.layers.Dense(
        7,
        activation="softmax"
    )

])


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


model.summary()


model_path = os.path.join(
    MODEL_DIR,
    "emotion_model_improved.keras"
)


callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=7,

        restore_best_weights=True

    ),

    tf.keras.callbacks.ModelCheckpoint(

        model_path,

        monitor="val_accuracy",

        save_best_only=True

    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=1e-6

    )

]


print(
    "\nStarting improved training..."
)


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=40,

    batch_size=64,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1

)


print(
    "\nEvaluating improved model..."
)


test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=1

)


print(
    "\n================================"
)

print(
    f"Improved Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)

print(
    "================================"
)


accuracy_path = os.path.join(
    RESULTS_DIR,
    "improved_accuracy.png"
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)


plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "EmotionVision AI - Improved Model Accuracy"
)

plt.legend()

plt.grid(True)


plt.savefig(
    accuracy_path,
    dpi=300
)

plt.close()


loss_path = os.path.join(
    RESULTS_DIR,
    "improved_loss.png"
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    history.history["loss"],
    label="Training Loss"
)


plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "EmotionVision AI - Improved Model Loss"
)

plt.legend()

plt.grid(True)


plt.savefig(
    loss_path,
    dpi=300
)

plt.close()


print(
    "\nImproved model saved to:"
)

print(
    model_path
)


print(
    "\nTraining graphs saved to:"
)

print(
    RESULTS_DIR
)
import os
import sys

import tensorflow as tf
import matplotlib.pyplot as plt


ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(ROOT_DIR)


from src.preprocess import load_dataset
from src.model import create_model


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


print("\nDataset information")

print(
    "Training images:",
    len(X_train)
)

print(
    "Testing images:",
    len(X_test)
)


print("\nCreating CNN model...")

model = create_model()

model.summary()


callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True

    ),

    tf.keras.callbacks.ModelCheckpoint(

        filepath=os.path.join(
            MODEL_DIR,
            "emotion_model.keras"
        ),

        monitor="val_accuracy",

        save_best_only=True

    )

]


print("\nStarting training...\n")


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=30,

    batch_size=64,

    callbacks=callbacks

)


print("\nEvaluating model...")


test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=1

)


print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)


accuracy_path = os.path.join(
    RESULTS_DIR,
    "accuracy.png"
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


plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "EmotionVision AI - Training Accuracy"
)

plt.legend()

plt.grid(True)


plt.savefig(
    accuracy_path
)

plt.close()


loss_path = os.path.join(
    RESULTS_DIR,
    "loss.png"
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


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "EmotionVision AI - Training Loss"
)

plt.legend()

plt.grid(True)


plt.savefig(
    loss_path
)

plt.close()


print("\nTraining completed!")

print(
    "Model saved in:",
    MODEL_DIR
)

print(
    "Graphs saved in:",
    RESULTS_DIR
)
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

try:
    import tensorflow as tf
    load_model = tf.keras.models.load_model
except (ImportError, AttributeError):
    from keras.models import load_model


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


TEST_PATH = os.path.join(
    ROOT_DIR,
    "dataset",
    "test"
)

MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "emotion_model_improved.keras"
)

RESULTS_DIR = os.path.join(
    ROOT_DIR,
    "results"
)


print("\nLoading test dataset...")

X_test, y_test = load_dataset(
    TEST_PATH
)

print(
    f"Test images: {len(X_test)}"
)


print("\nLoading improved CNN model...")

model = load_model(
    MODEL_PATH
)


print("\nGenerating predictions...")

predictions = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)


print("\nImproved Model Classification Report")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    target_names=EMOTIONS
)

print(report)


report_path = os.path.join(
    RESULTS_DIR,
    "improved_classification_report.txt"
)

with open(
    report_path,
    "w"
) as file:
    file.write(report)


print(
    f"\nClassification report saved to:"
    f"\n{report_path}"
)


cm = confusion_matrix(
    y_test,
    y_pred
)


plt.figure(
    figsize=(9, 7)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=EMOTIONS,
    yticklabels=EMOTIONS
)

plt.xlabel(
    "Predicted Emotion"
)

plt.ylabel(
    "Actual Emotion"
)

plt.title(
    "EmotionVision AI - Improved Model Confusion Matrix"
)

plt.tight_layout()


cm_path = os.path.join(
    RESULTS_DIR,
    "improved_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.show()


print(
    f"\nConfusion matrix saved to:"
    f"\n{cm_path}"
)

print("\nImproved model evaluation completed successfully!")
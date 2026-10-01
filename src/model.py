import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models


def create_model():

    model = models.Sequential([

        layers.Input(
            shape=(48, 48, 1)
        ),

        layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Dropout(0.25),


        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Dropout(0.25),


        layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Dropout(0.25),


        layers.Flatten(),

        layers.Dense(
            256,
            activation="relu"
        ),

        layers.Dropout(0.5),

        layers.Dense(
            7,
            activation="softmax"
        )

    ])


    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )


    return model
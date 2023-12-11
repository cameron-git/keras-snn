import keras
from . import utils

BATCH_SIZE = 128

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

inputs = keras.Input(shape=(28, 28, 1), name="digits")
x = keras.layers.Conv2D(
    64, kernel_size=3, strides=2, padding="same", activation="relu"
)(inputs)
x = keras.layers.Conv2D(
    64, kernel_size=3, strides=2, padding="same", activation="relu"
)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(512, activation="relu")(x)
x = keras.layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=x, name="mnist_model")
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

print